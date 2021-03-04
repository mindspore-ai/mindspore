/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
package com.mindspore.classificationforpet.widget;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.classificationforpet.R;
import com.mindspore.classificationforpet.gallery.classify.ImageTrackingMobile;
import com.mindspore.classificationforpet.gallery.classify.RecognitionImageBean;
import com.mindspore.classificationforpet.gallery.classify.TrackingMobile;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The main interface of camera preview.
 * Using Camera 2 API.
 */
public class CameraActivity extends AppCompatActivity {
    private static final String TAG = "CameraActivity";
    private static final String IMAGE_SCENE_MS = "model/mobilenetv2.ms";

    private String filePath;
    private boolean isPetModel;
    private TrackingMobile trackingMobile;
    private ImageTrackingMobile imageTrackingMobile;

    private TextView resultText;
    private List<RecognitionImageBean> recognitionObjectBeanList;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        filePath = getIntent().getStringExtra("FILEPATH");
        isPetModel = getIntent().getBooleanExtra("ISHASPetMODELFILE", false);
        resultText = findViewById(R.id.textResult);

        if (isPetModel) {
            trackingMobile = new TrackingMobile(this);
            boolean ret = trackingMobile.loadModelFromBuf(filePath);
            if (!ret) {
                Log.e(TAG, "Load model error.");
                return;
            }
        } else {
            imageTrackingMobile = new ImageTrackingMobile(this);
            boolean ret = imageTrackingMobile.loadModelFromBuf(IMAGE_SCENE_MS);
            if (!ret) {
                Log.e(TAG, "Load model error.");
                return;
            }
        }
        addCameraFragment();
    }

    protected void addCameraFragment() {
        CameraFragment cameraFragment = CameraFragment.newInstance(bitmap -> {
            runOnUiThread(() -> initMindspore(bitmap));
        });

        getSupportFragmentManager().beginTransaction()
                .replace(R.id.container, cameraFragment)
                .commitAllowingStateLoss();
    }


    private void initMindspore(Bitmap bitmap) {
        // run net.
        if (isPetModel) {
            long startTime = System.currentTimeMillis();
            String result = trackingMobile.MindSpore_runnet(bitmap);
            long endTime = System.currentTimeMillis();
            String[] IMAGECONTENT = getResources().getStringArray(R.array.image_category_pet);
            int nameIndex = Integer.parseInt(result);
            resultText.setText(IMAGECONTENT[nameIndex]);
            Log.d(TAG, "RUNNET CONSUMING：" + (endTime - startTime) + "ms");
            Log.d(TAG, "result：" + result);
        } else {
            if (recognitionObjectBeanList != null) {
                recognitionObjectBeanList.clear();
            } else {
                recognitionObjectBeanList = new ArrayList<>();
            }

            long startTime = System.currentTimeMillis();
            String result = imageTrackingMobile.MindSpore_runnet(bitmap);
            long endTime = System.currentTimeMillis();
            Log.d(TAG, "RUNNET CONSUMING：" + (endTime - startTime) + "ms");
            Log.d(TAG, "result：" + result);
            String[] IMAGECONTENT = getResources().getStringArray(R.array.image_category);
            if (!TextUtils.isEmpty(result)) {
                String[] resultArray = result.split(";");
                for (String singleRecognitionResult : resultArray) {
                    String[] singleResult = singleRecognitionResult.split(":");
                    int nameIndex = Integer.parseInt(singleResult[0]);
                    float score = Float.parseFloat(singleResult[1]);
                    if (score > 0.5) {
                        recognitionObjectBeanList.add(new RecognitionImageBean(IMAGECONTENT[nameIndex], score));
                    }
                }
                Collections.sort(recognitionObjectBeanList, (t1, t2) -> Float.compare(t2.getScore(), t1.getScore()));
                showResultsInBottomSheet(recognitionObjectBeanList, (endTime - startTime) + "ms");
            }
        }
    }


    @UiThread
    protected void showResultsInBottomSheet(List<RecognitionImageBean> list, String time) {
        if (list == null || list.size() < 1) {
            return;
        }
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < list.size(); i++) {
            RecognitionImageBean bean = list.get(i);
            stringBuilder.append(bean.getName()).append("\r:\r").append(String.format("%.2f", (100 * bean.getScore())) + "%").append("\r\n");
            if (i > 3) { // set maximum display is 3.
                break;
            }
        }
        resultText.setText(stringBuilder);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (trackingMobile != null) {
            trackingMobile.unloadModel();
        }
        if (imageTrackingMobile != null) {
            imageTrackingMobile.unloadModel();
        }
    }
}
