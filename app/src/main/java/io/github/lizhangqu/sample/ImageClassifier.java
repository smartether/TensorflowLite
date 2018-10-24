/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package io.github.lizhangqu.sample;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.tensorflow.lite.Interpreter;


//open cv
import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.util.*;

/** Classifies images with Tensorflow Lite. */
public class ImageClassifier {

  /** Tag for the {@link Log}. */
  private static final String TAG = "TfLiteCameraDemo";
  private static final String TAG1 = "HandRaised";
  private static final String TAG2 = "Pos";

  /** Name of the model file stored in Assets. */
  private static final String MODEL_PATH = "mobilenet_quant_v1_224.tflite";
  private static final String OPENPOSE_MODEL_PATH = "graph_1_368_368_3.tflite";

  /** Name of the label file stored in Assets. */
  private static final String LABEL_PATH = "labels.txt";

  /** Number of results to show in the UI. */
  private static final int RESULTS_TO_SHOW = 3;

  /** Dimensions of inputs. */
  private static final int DIM_BATCH_SIZE = 1;

  private static final int DIM_PIXEL_SIZE = 3 * 4;

  static final int DIM_IMG_SIZE_X = 368;//224;
  static final int DIM_IMG_SIZE_Y = 368;//224;

  /* Preallocated buffers for storing image data in. */
  private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  private Interpreter tflite;

  /** Labels corresponding to the output of the vision model. */
  private List<String> labelList;

  /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
  private ByteBuffer imgData = null;

  /** An array to hold inference results, to be feed into Tensorflow Lite as outputs. */
  private float[][][][] labelProbArray = null;

  private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
      new PriorityQueue<>(
          RESULTS_TO_SHOW,
          new Comparator<Map.Entry<String, Float>>() {
            @Override
            public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
              return (o1.getValue()).compareTo(o2.getValue());
            }
          });


  /** Initializes an {@code ImageClassifier}. */
  ImageClassifier(Activity activity) throws IOException {
    //CORRECT
    tflite = new Interpreter(loadOpenPoseModelFile(activity));
    //labelList = loadLabelList(activity);
    imgData =
        ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
    imgData.order(ByteOrder.nativeOrder());
    //labelProbArray = new byte[1][labelList.size()];
    labelProbArray = new float[1][46][46][57];
    Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
  }

  boolean isPrinted = false;
  int frameCount = 0;
  /** Classifies a frame from the preview stream. */
  String classifyFrame(Bitmap bitmap) {
    if (tflite == null) {
      Log.e(TAG, "Image classifier has not been initialized; Skipped.");
      return "Uninitialized Classifier.";
    }
    convertBitmapToByteBuffer(bitmap);
    // Here's where the magic happens!!!
    long endTime=0;
    long startTime = 0;
    startTime = SystemClock.uptimeMillis();

    tflite.run(imgData, labelProbArray);



      endTime = SystemClock.uptimeMillis();
      Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));
      startTime = SystemClock.uptimeMillis();

    ArrayList<Point> points = new ArrayList();
    Mat mat = MatOfFloat.zeros(new Size(46,46), CvType.CV_32F);
    for(int heatIdx=0;heatIdx<19;heatIdx++) {
      StringBuffer s = new StringBuffer("$$ ");
      for (int i = 0; i < 46; i++) {
        for (int j = 0; j < 46; j++) {
          float p =  labelProbArray[0][i][j][heatIdx];
            mat.put(i, j, p);
//          if( p>0){
//            s.append(p);
//            s.append(" ");
//            mat.put(i, j, p);
//          }
//          else{
//            mat.put(i,j,0);
//            s.append(0);
//            s.append(" ");
//          }
        }
      }

      //Log.d(TAG, "$$ p:" + s.toString());

      endTime = SystemClock.uptimeMillis();
      Log.d(TAG, "Timecost to run MatOfFloat transform: " + Long.toString(endTime - startTime));
      startTime = SystemClock.uptimeMillis();

      //Log.d(TAG, "mat width:" + mat.size().width + " height:"+ mat.size().height);
      Core.MinMaxLocResult mm = Core.minMaxLoc(mat);
      Point p = new Point(-1,-1);
      if (mm.maxVal>0.01f) {
        p = mm.maxLoc;
      }
      points.add(p);
      Log.d(TAG2, heatIdx + " max:" + p);// + " min:" + min );
    }
    if(points.get(4).x > 0 && points.get(2).x > 0 && points.get(4).y < points.get(2).y){
        Log.d(TAG1,   "$$ hand Raised");
    }
    //Mat result = mat.reshape(1, 19);


    endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to run MinMaxLocResult: " + Long.toString(endTime - startTime));
    String textToShow = "";//printTopKLabels();
    textToShow = Long.toString(endTime - startTime) + "ms" + textToShow;
    return textToShow;
  }

  /** Closes tflite to release resources. */
  public void close() {
    tflite.close();
    tflite = null;
  }

  /** Reads label list from Assets. */
  private List<String> loadLabelList(Activity activity) throws IOException {
    List<String> labelList = new ArrayList<String>();
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(activity.getAssets().open(LABEL_PATH)));
    String line;
    while ((line = reader.readLine()) != null) {
      labelList.add(line);
    }
    reader.close();
    return labelList;
  }

  /** Memory-map the model file in Assets. */
  private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /** Memory-map the model file in Assets. */
  private MappedByteBuffer loadOpenPoseModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(OPENPOSE_MODEL_PATH);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /** Writes Image data into a {@code ByteBuffer}. */
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // Convert the image to floating point.

    long startTime = SystemClock.uptimeMillis();
    int pixel = 0;
    float fact =  1;// (float)1 / (float)255;
    //Log.d(TAG, "$$ fact:" + fact);
/*
    MatOfInt imageMat = new MatOfInt(intValues);
    Mat imageMatf = Mat.zeros(368,368, CvType.CV_32F);
    imageMat.convertTo(imageMatf, CvType.CV_32F, fact, 0);
    float[] fArray  = new float[368 * 368 * 3];
    for(int i=0,c=fArray.length;i<c;i++) {
      imgData.putFloat(fArray[i]);
    }
*/


    for (int i = 0; i < DIM_IMG_SIZE_X - 1; ++i) {
      for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
        final int val = intValues[pixel++];
        imgData.putFloat(((val >> 16) & 0xFF) * fact);
        imgData.putFloat(((val >> 8) & 0xFF) * fact);
        imgData.putFloat((val & 0xFF) * fact);
      }
    }

    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
  }

  /** Prints top-K labels, to be shown in UI as the results. */
  private String printTopKLabels() {
    for (int i = 0; i < labelList.size(); ++i) {
      sortedLabels.add(new AbstractMap.SimpleEntry<>("",1.0f));
          //new AbstractMap.SimpleEntry<>(labelList.get(i), (labelProbArray[0][i] & 0xff) / 255.0f));
      if (sortedLabels.size() > RESULTS_TO_SHOW) {
        sortedLabels.poll();
      }
    }
    String textToShow = "";
    final int size = sortedLabels.size();
    for (int i = 0; i < size; ++i) {
      Map.Entry<String, Float> label = sortedLabels.poll();
      textToShow = "\n" + label.getKey() + ":" + Float.toString(label.getValue()) + textToShow;
    }
    return textToShow;
  }
}
