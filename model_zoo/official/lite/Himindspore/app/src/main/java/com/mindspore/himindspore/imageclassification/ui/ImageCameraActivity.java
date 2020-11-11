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

package com.mindspore.himindspore.imageclassification.ui;

import android.graphics.Color;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.UiThread;
import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.himindspore.R;
import com.mindspore.himindspore.camera.CameraPreview;
import com.mindspore.himindspore.imageclassification.bean.RecognitionImageBean;
import com.mindspore.himindspore.imageclassification.help.GarbageTrackingMobile;
import com.mindspore.himindspore.imageclassification.help.ImageTrackingMobile;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * The main interface of camera preview.
 * Using Camera 2 API.
 */
public class ImageCameraActivity extends AppCompatActivity implements CameraPreview.RecognitionDataCallBack {
    private static final String TAG = "ImageCameraActivity";
    public static final String OPEN_TYPE = "OPEN_TYPE";
    public static final int TYPE_DEMO = 1;
    public static final int TYPE_CUSTOM = 2;

    private int enterType;

    private LinearLayout bottomLayout;

    private List<RecognitionImageBean> recognitionObjectBeanList;

    private CameraPreview cameraPreview;

    private ImageTrackingMobile mTrackingMobile;

    private GarbageTrackingMobile garbageTrackingMobile;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_camera);
        enterType = getIntent().getIntExtra(OPEN_TYPE, TYPE_DEMO);

        cameraPreview = findViewById(R.id.image_camera_preview);
        bottomLayout = findViewById(R.id.layout_bottom_content);
        cameraPreview.setVisibility(View.VISIBLE);
        init();
    }

    private void init() {
        if (enterType == TYPE_DEMO) {
            mTrackingMobile = new ImageTrackingMobile(this);
            String modelPath = "model/mobilenetv2.ms";
            boolean ret = mTrackingMobile.loadModelFromBuf(modelPath);
            Log.d(TAG, "Loading model return value: " + ret);
        } else {
            garbageTrackingMobile = new GarbageTrackingMobile(this);
            String garbageModelPath = "model/garbage_mobilenetv2.ms";
            boolean garbageRet = garbageTrackingMobile.loadModelFromBuf(garbageModelPath);
            Log.d(TAG, "Garbage Loading model return value: " + garbageRet);
        }
        cameraPreview.addImageRecognitionDataCallBack(this);
    }


    @Override
    protected void onResume() {
        super.onResume();
        if (enterType == TYPE_DEMO) {
            cameraPreview.onResume(this, CameraPreview.OPEN_TYPE_IMAGE, mTrackingMobile);
        } else {
            cameraPreview.onResume(this, CameraPreview.OPEN_TYPE_IMAGE_CUSTOM, garbageTrackingMobile);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        cameraPreview.onPause();
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (mTrackingMobile != null) {
            boolean ret = mTrackingMobile.unloadModel();
            Log.d(TAG, "Unload model return value: " + ret);
        }
        if (garbageTrackingMobile != null) {
            boolean ret = garbageTrackingMobile.unloadModel();
            Log.d(TAG, "garbageTrackingMobile Unload model return value: " + ret);
        }
    }


    @Override
    public void onRecognitionDataCallBack(final String result, final String time) {
        if (enterType == TYPE_DEMO) {
            if (recognitionObjectBeanList != null) {
                recognitionObjectBeanList.clear();
            } else {
                recognitionObjectBeanList = new ArrayList<>();
            }

            if (!result.equals("")) {
                String[] resultArray = result.split(";");
                for (String singleRecognitionResult : resultArray) {
                    String[] singleResult = singleRecognitionResult.split(":");
                    float score = Float.parseFloat(singleResult[1]);
                    if (score > 0.5) {
                        recognitionObjectBeanList.add(new RecognitionImageBean(singleResult[0], score));
                    }
                }
                Collections.sort(recognitionObjectBeanList, new Comparator<RecognitionImageBean>() {
                    @Override
                    public int compare(RecognitionImageBean t1, RecognitionImageBean t2) {
                        return Float.compare(t2.getScore(), t1.getScore());
                    }
                });
            }

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    showResultsInBottomSheet(recognitionObjectBeanList, time);
                }
            });
        } else {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    showResultsInBottomSheetGarbage(result, time);
                }
            });
        }
    }

    @UiThread
    protected void showResultsInBottomSheet(List<RecognitionImageBean> list, String time) {
        bottomLayout.removeAllViews();
        if (list != null && list.size() > 0) {
            int classNum = 0;
            for (RecognitionImageBean bean : list) {
                classNum++;
                HorTextView horTextView = new HorTextView(this);
                horTextView.setLeftTitle(bean.getName());
                horTextView.setRightContent(String.format("%.2f", (100 * bean.getScore())) + "%");
                horTextView.setBottomLineVisible(View.VISIBLE);
                if (classNum == 1) {
                    horTextView.getTvLeftTitle().setTextColor(getResources().getColor(R.color.text_blue));
                    horTextView.getTvRightContent().setTextColor(getResources().getColor(R.color.text_blue));
                } else {
                    horTextView.getTvLeftTitle().setTextColor(getResources().getColor(R.color.white));
                    horTextView.getTvRightContent().setTextColor(getResources().getColor(R.color.white));
                }
                bottomLayout.addView(horTextView);
                if (classNum > 4) { // set maximum display is 5.
                    break;
                }
            }
            HorTextView horTextView = new HorTextView(this);
            horTextView.setLeftTitle(getResources().getString(R.string.title_time));
            horTextView.setRightContent(time);
            horTextView.setBottomLineVisible(View.INVISIBLE);
            horTextView.getTvLeftTitle().setTextColor(getResources().getColor(R.color.text_blue));
            horTextView.getTvRightContent().setTextColor(getResources().getColor(R.color.text_blue));
            bottomLayout.addView(horTextView);
        } else {
            showLoadView();
        }
    }

    @UiThread
    protected void showResultsInBottomSheetGarbage(String result, String time) {
        bottomLayout.removeAllViews();
        if (!TextUtils.isEmpty(result)) {
            HorTextView horTextView = new HorTextView(this);
            horTextView.setLeftTitle(result);
            horTextView.setBottomLineVisible(View.VISIBLE);
            bottomLayout.addView(horTextView);
        } else {
            showLoadView();
        }
    }

    private void showLoadView() {
        TextView textView = new TextView(this);
        textView.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        textView.setText("Keep moving.");
        textView.setGravity(Gravity.CENTER);
        textView.setTextColor(Color.WHITE);
        textView.setTextSize(30);
        bottomLayout.addView(textView);
    }
}
