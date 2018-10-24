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
import android.os.Bundle;

import org.opencv.android.InstallCallbackInterface;
import org.opencv.android.LoaderCallbackInterface;

/** Main {@code Activity} class for the Camera app. */
public class CameraActivity extends Activity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_camera);

/*
    org.opencv.android.OpenCVLoader.initAsync("3.4.1", getApplicationContext(), new LoaderCallbackInterface() {
      @Override
      public void onManagerConnected(int i) {
        //if (null == savedInstanceState) {

       // }
      }

      @Override
      public void onPackageInstall(int i, InstallCallbackInterface installCallbackInterface) {

      }
    });
*/
    getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    org.opencv.android.OpenCVLoader.initDebug();

    getFragmentManager()
            .beginTransaction()
            .replace(R.id.container, Camera2BasicFragment.newInstance())
            .commit();

  }
}
