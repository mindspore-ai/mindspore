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

package com.mindspore.imageobject.imageclassification.ui;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.UiThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.alibaba.android.arouter.launcher.ARouter;
import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.common.utils.Utils;
import com.mindspore.customview.dialog.NoticeDialog;
import com.mindspore.imageobject.R;
import com.mindspore.imageobject.camera.CameraPreview;
import com.mindspore.imageobject.imageclassification.bean.RecognitionImageBean;
import com.mindspore.imageobject.imageclassification.help.ImageTrackingMobile;

/**
 * The main interface of camera preview.
 * Using Camera 2 API.
 */
@Route(path = "/imageobject/ImageCameraActivity")
public class ImageCameraActivity extends AppCompatActivity implements CameraPreview.RecognitionDataCallBack {
    private static final String TAG = "ImageCameraActivity";

    private static final String IMAGE_SCENE_MS = "model/mobilenetv2.ms";
    private LinearLayout bottomLayout;
    private CameraPreview cameraPreview;
    private ImageTrackingMobile imageTrackingMobile;
    private RecognitionImageBean bean;
    private NoticeDialog noticeDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //inject
        ARouter.getInstance().inject(this);
        setContentView(R.layout.activity_image_camera);
        cameraPreview = findViewById(R.id.image_camera_preview);
        bottomLayout = findViewById(R.id.layout_bottom_content);
        cameraPreview.setVisibility(View.VISIBLE);
        init();
    }

    private void init() {
        Toolbar mToolbar = findViewById(R.id.image_camera_toolbar);
        mToolbar.setTitle(getString(R.string.image_camera_title));
        imageTrackingMobile = new ImageTrackingMobile(this);
        boolean ret = imageTrackingMobile.loadModelFromBuf(IMAGE_SCENE_MS);
        Log.d(TAG, "Loading model return value: " + ret);
        cameraPreview.addImageRecognitionDataCallBack(this);
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu info");
        getMenuInflater().inflate(R.menu.menu_setting_app, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int itemId = item.getItemId();
        if (itemId == R.id.item_help) {
            showHelpDialog();
        } else if (itemId == R.id.item_more) {
            Utils.openBrowser(this, MSLinkUtils.HELP_IMAGE_CLASSIFICATION);

        }
        return super.onOptionsItemSelected(item);
    }

    private void showHelpDialog() {
        noticeDialog = new NoticeDialog(this);
        noticeDialog.setTitleString(getString(R.string.explain_title));
        noticeDialog.setContentString(getString(R.string.explain_image_classification));
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

    @Override
    protected void onStop() {
        super.onStop();
        if (imageTrackingMobile != null) {
            boolean ret = imageTrackingMobile.unloadModel();
            Log.d(TAG, "imageTrackingMobile Unload model return value: " + ret);
        }
    }

    @Override
    public void onRecognitionBitmapCallBack(Bitmap bitmap) {
        long startTime = System.currentTimeMillis();
        String result = imageTrackingMobile.MindSpore_runnet(bitmap);
        long endTime = System.currentTimeMillis();
        onRecognitionDataCallBack(result, (endTime - startTime) + "ms ");
        if (!bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }


    public void onRecognitionDataCallBack(final String result, final String time) {
            if (TextUtils.isEmpty(result)) {
                return;
            }
            int maxIndex = Integer.parseInt(result);
            String[] IMAGECONTENT = getResources().getStringArray(R.array.image_category_old);
            runOnUiThread(() -> showResultsInImage(IMAGECONTENT[maxIndex], time));
    }

    @UiThread
    protected void showResultsInImage(String result, String time) {
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
        textView.setText(getResources().getString(R.string.title_keep));
        textView.setGravity(Gravity.CENTER);
        textView.setTextColor(Color.BLACK);
        textView.setTextSize(30);
        bottomLayout.addView(textView);
    }

}
