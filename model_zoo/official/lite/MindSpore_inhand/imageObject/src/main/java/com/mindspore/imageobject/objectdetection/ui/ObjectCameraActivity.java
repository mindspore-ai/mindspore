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
import android.view.Menu;
import android.view.MenuItem;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.common.utils.Utils;
import com.mindspore.customview.dialog.NoticeDialog;
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

    private NoticeDialog noticeDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_object_camera);
        Log.i(TAG, "onCreate ObjectCameraActivity info");
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

        Log.i(TAG, "init ObjectPhotoActivity info");
        Toolbar mToolbar = findViewById(R.id.object_camera_toolbar);
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_setting_app, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int itemId = item.getItemId();
        if (itemId == R.id.item_help) {
            showHelpDialog();
        } else if (itemId == R.id.item_more) {
            Utils.openBrowser(this, MSLinkUtils.HELP_CAMERA_DETECTION);
        }
        return super.onOptionsItemSelected(item);
    }

    private void showHelpDialog() {
        noticeDialog = new NoticeDialog(this);
        noticeDialog.setTitleString(getString(R.string.explain_title));
        noticeDialog.setContentString(getString(R.string.explain_camera_detection));
        noticeDialog.setYesOnclickListener(() -> {
            noticeDialog.dismiss();
        });
        noticeDialog.show();
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
