/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.scene.widget;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.UiThread;
import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.scene.R;
import com.mindspore.scene.gallery.classify.RecognitionObjectBean;

/**
 * The main interface of camera preview.
 * Using Camera 2 API.
 */
public class CameraActivity extends AppCompatActivity {
    private static final String TAG = "CameraActivity";

    private static final String BUNDLE_FRAGMENTS_KEY = "android:support:fragments";

    private static final int PERMISSIONS_REQUEST = 1;

    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;

    private LinearLayout bottomLayout;

    private RecognitionObjectBean recognitionObjectBean;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        Log.d(TAG, "onCreate");

        if (savedInstanceState != null && this.clearFragmentsTag()) {
            // Clear the state of the fragment when rebuilding.
            savedInstanceState.remove(BUNDLE_FRAGMENTS_KEY);
        }

        super.onCreate(null);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_camera);

        if (hasPermission()) {
            setFragment();
        } else {
            requestPermission();
        }

        bottomLayout = findViewById(R.id.layout_bottom_content);
    }

    @Override
    public void onRequestPermissionsResult(final int requestCode, final String[] permissions,
                                           final int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST) {
            if (allPermissionsGranted(grantResults)) {
                setFragment();
            } else {
                requestPermission();
            }
        }
    }

    private static boolean allPermissionsGranted(final int[] grantResults) {
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(CameraActivity.this, "Camera permission is required for this demo", Toast.LENGTH_LONG)
                        .show();
            }
            requestPermissions(new String[]{PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
        }
    }


    protected void setFragment() {
        CameraFragment cameraFragment = CameraFragment.newInstance(new CameraFragment.RecognitionDataCallBack() {
            @Override
            public void onRecognitionDataCallBack(String result, final String time) {
                dealRecognitionData(result, time);
            }
        });

        getSupportFragmentManager().beginTransaction()
                .replace(R.id.container, cameraFragment)
                .commitAllowingStateLoss();
    }


    private void dealRecognitionData(String result, final String time) {
        if (!result.equals("") && result.contains(":")) {
            String[] resultArray = result.split(":");
            recognitionObjectBean = new RecognitionObjectBean(resultArray[0], Float.valueOf(resultArray[1]));
        }

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                showResultsInBottomSheet(time);
            }
        });
    }

    @UiThread
    protected void showResultsInBottomSheet(String time) {
        bottomLayout.removeAllViews();
        if (recognitionObjectBean != null) {
            HorTextView horTextView = new HorTextView(this);
            horTextView.setLeftTitle(recognitionObjectBean.getName() + ":");
            horTextView.setRightContent(String.format("%.2f", (100 * recognitionObjectBean.getScore())) + "%");
            horTextView.setBottomLineVisible(View.VISIBLE);
            bottomLayout.addView(horTextView);

            HorTextView horTimeView = new HorTextView(this);
            horTimeView.setLeftTitle("Inference Time：");
            horTimeView.setRightContent(time);
            horTimeView.setBottomLineVisible(View.INVISIBLE);
            bottomLayout.addView(horTimeView);
        } else {
            TextView textView = new TextView(this);
            textView.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
            textView.setText("Keep moving.");
            textView.setGravity(Gravity.CENTER);
            textView.setTextColor(Color.BLACK);
            textView.setTextSize(30);
            bottomLayout.addView(textView);
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        if (outState != null && this.clearFragmentsTag()) {
            outState.remove(BUNDLE_FRAGMENTS_KEY);
        }
    }

    protected boolean clearFragmentsTag() {
        return true;
    }
}
