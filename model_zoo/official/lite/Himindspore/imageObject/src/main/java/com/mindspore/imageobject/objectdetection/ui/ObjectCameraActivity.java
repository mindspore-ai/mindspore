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
package com.mindspore.imageobject.objectdetection.ui;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.mindspore.imageobject.R;
import com.mindspore.imageobject.camera.CameraPreview;
import com.mindspore.imageobject.objectdetection.bean.RecognitionObjectBean;
import com.mindspore.imageobject.objectdetection.help.ObjectTrackingMobile;

import java.io.FileNotFoundException;
import java.util.List;

import static com.mindspore.imageobject.objectdetection.bean.RecognitionObjectBean.getRecognitionList;


/**
 * main page of entrance
 * <p>
 * Pass in pictures to JNI, test mindspore model, load reasoning, etc
 */
@Route(path = "/imageobject/ObjectCameraActivity")
public class ObjectCameraActivity extends AppCompatActivity implements CameraPreview.RecognitionDataCallBack {

    private final String TAG = "ObjectCameraActivity";

    private CameraPreview cameraPreview;

    private ObjectTrackingMobile mTrackingMobile;

    private ObjectRectView mObjectRectView;

    private List<RecognitionObjectBean> recognitionObjectBeanList;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_object_camera);

        cameraPreview = findViewById(R.id.camera_preview);
        mObjectRectView = findViewById(R.id.objRectView);

        init();
    }

    private void init() {
        try {
            mTrackingMobile = new ObjectTrackingMobile(this);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        boolean ret = mTrackingMobile.loadModelFromBuf(getAssets());
        Log.d(TAG, "TrackingMobile loadModelFromBuf: " + ret);

        cameraPreview.addImageRecognitionDataCallBack(this);
    }


    @Override
    protected void onResume() {
        super.onResume();
        cameraPreview.onResume(this);

    }

    @Override
    protected void onPause() {
        super.onPause();
        cameraPreview.onPause();
    }

    public void onRecognitionDataCallBack(String result) {
        if (TextUtils.isEmpty(result)) {
            mObjectRectView.clearCanvas();
            return;
        }
        Log.d(TAG, result);
        recognitionObjectBeanList = getRecognitionList(result);
        mObjectRectView.setInfo(recognitionObjectBeanList);
    }

    @Override
    public void onRecognitionBitmapCallBack(Bitmap bitmap) {
        long startTime = System.currentTimeMillis();
        String result = mTrackingMobile.MindSpore_runnet(bitmap);
        long endTime = System.currentTimeMillis();
        Log.d(TAG,"TrackingMobile inferenceTime:"+(endTime - startTime) + "ms ");
        onRecognitionDataCallBack(result);
        if (!bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }
}
