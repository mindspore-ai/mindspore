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
package com.mindspore.hms.gesturerecognition;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.alibaba.android.arouter.launcher.ARouter;
import com.bumptech.glide.Glide;
import com.huawei.hmf.tasks.Task;
import com.huawei.hms.mlsdk.common.MLFrame;
import com.huawei.hms.mlsdk.gesture.MLGesture;
import com.huawei.hms.mlsdk.gesture.MLGestureAnalyzer;
import com.huawei.hms.mlsdk.gesture.MLGestureAnalyzerFactory;
import com.huawei.hms.mlsdk.gesture.MLGestureAnalyzerSetting;
import com.mindspore.hms.BitmapUtils;
import com.mindspore.hms.R;
import com.mindspore.hms.camera.GraphicOverlay;
import com.mindspore.hms.camera.HandGestureGraphic;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;

@Route(path = "/hms/StillHandGestureAnalyseActivity")
public class StillHandGestureAnalyseActivity extends AppCompatActivity {

    private static final String TAG = "TextRecognitionActivity";

    private GraphicOverlay mGraphicOverlay;
    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int RC_CHOOSE_CAMERA = 2;

    private boolean isPreViewShow = false;

    private ImageView imgPreview;
    private TextView textOriginImage;
    private Uri imageUri;

    private Bitmap originBitmap;
    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    private boolean isLandScape;
    private MLGestureAnalyzer mAnalyzer;
    private MLFrame frame;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gesture_recognition);
        init();
    }

    private void init() {
        mGraphicOverlay = findViewById(R.id.skeleton_previewOverlay);
        imgPreview = findViewById(R.id.img_origin);
        textOriginImage = findViewById(R.id.tv_image);
        Toolbar mToolbar = findViewById(R.id.activity_toolbar);
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
    }

    public void onClickPhoto(View view) {
        openGallay(RC_CHOOSE_PHOTO);
        textOriginImage.setVisibility(View.GONE);
    }

    public void onClickCamera(View view) {
        openCamera();
        textOriginImage.setVisibility(View.GONE);
    }


    public void onClickRealTime(View view) {
        ARouter.getInstance().build("/hms/LiveHandGestureAnalyseActivity").navigation();
    }

    private void openGallay(int request) {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, request);
    }

    private void openCamera() {
        Intent intentToTakePhoto = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        String mTempPhotoPath = Environment.getExternalStorageDirectory() + File.separator + "photo.jpeg";
        imageUri = FileProvider.getUriForFile(this, getApplicationContext().getPackageName() + ".fileprovider", new File(mTempPhotoPath));
        intentToTakePhoto.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        startActivityForResult(intentToTakePhoto, RC_CHOOSE_CAMERA);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (RC_CHOOSE_PHOTO == requestCode) {
                if (null != data && null != data.getData()) {
                    this.imageUri = data.getData();
                    showOriginImage();
                } else {
                    finish();
                }
            } else if (RC_CHOOSE_CAMERA == requestCode) {
                showOriginCamera();
            } else {
                textOriginImage.setVisibility(!isPreViewShow ? View.VISIBLE : View.GONE);
            }
        }
    }

    private void showOriginImage() {
        File file = BitmapUtils.getFileFromMediaUri(this, imageUri);
        Bitmap photoBmp = BitmapUtils.getBitmapFormUri(this, Uri.fromFile(file));
        int degree = BitmapUtils.getBitmapDegree(file.getAbsolutePath());
        originBitmap = BitmapUtils.rotateBitmapByDegree(photoBmp, degree).copy(Bitmap.Config.ARGB_8888, true);
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
            isPreViewShow = true;
            analyzerAsync();
        } else {
            isPreViewShow = false;
        }
    }

    private void createFrame() {
        // Gets the targeted width / height, only portrait.
        int maxHeight = ((View) imgPreview.getParent()).getHeight();
        int targetWidth = ((View) imgPreview.getParent()).getWidth();
        // Determine how much to scale down the image.
        float scaleFactor =
                Math.max(
                        (float) originBitmap.getWidth() / (float) targetWidth,
                        (float) originBitmap.getHeight() / (float) maxHeight);

        Bitmap resizedBitmap =
                Bitmap.createScaledBitmap(
                        originBitmap,
                        (int) (originBitmap.getWidth() / scaleFactor),
                        (int) (originBitmap.getHeight() / scaleFactor),
                        true);

        frame = new MLFrame.Creator().setBitmap(resizedBitmap).create();

        MLGestureAnalyzerSetting setting =
                new MLGestureAnalyzerSetting.Factory()
                        .create();
        mAnalyzer = MLGestureAnalyzerFactory.getInstance().getGestureAnalyzer(setting);
    }

    /**
     * Asynchronous analyse.
     */
    private void analyzerAsync() {
        createFrame();
        Task<List<MLGesture>> task = mAnalyzer.asyncAnalyseFrame(frame);
        task.addOnSuccessListener(results -> {
            // Detection success.
            if (results != null && !results.isEmpty()) {
                processSuccess(results);
            } else {
                processFailure("async analyzer result is null.");
            }
        }).addOnFailureListener(e -> {
            // Detection failure.
            processFailure(e.getMessage());
        });
    }

    private void processSuccess(List<MLGesture> results) {
        mGraphicOverlay.clear();
        HandGestureGraphic handGraphic = new HandGestureGraphic(mGraphicOverlay, results);
        mGraphicOverlay.add(handGraphic);
    }

    private void processFailure(String str) {
        Log.e(TAG, str);
        Toast.makeText(this, R.string.gesture_detected_failure, Toast.LENGTH_SHORT).show();
    }

    private void showOriginCamera() {
        try {
            Pair<Integer, Integer> targetedSize = this.getTargetSize();
            int targetWidth = targetedSize.first;
            int maxHeight = targetedSize.second;
            Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
            originBitmap = BitmapUtils.zoomImage(bitmap, targetWidth, maxHeight);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        // Determine how much to scale down the image.
        Log.e(TAG, "resized image size width:" + originBitmap.getWidth() + ",height: " + originBitmap.getHeight());
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
            isPreViewShow = true;
            analyzerAsync();
        } else {
            isPreViewShow = false;
        }
    }

    private Pair<Integer, Integer> getTargetSize() {
        Integer targetWidth;
        Integer targetHeight;
        Integer maxWidth = this.getMaxWidthOfImage();
        Integer maxHeight = this.getMaxHeightOfImage();
        targetWidth = this.isLandScape ? maxHeight : maxWidth;
        targetHeight = this.isLandScape ? maxWidth : maxHeight;
        Log.i(TAG, "height:" + targetHeight + ",width:" + targetWidth);
        return new Pair<>(targetWidth, targetHeight);
    }

    private Integer getMaxWidthOfImage() {
        if (this.maxWidthOfImage == null) {
            if (this.isLandScape) {
                this.maxWidthOfImage = ((View) this.imgPreview.getParent()).getHeight();
            } else {
                this.maxWidthOfImage = ((View) this.imgPreview.getParent()).getWidth();
            }
        }
        return this.maxWidthOfImage;
    }

    private Integer getMaxHeightOfImage() {
        if (this.maxHeightOfImage == null) {
            if (this.isLandScape) {
                this.maxHeightOfImage = ((View) this.imgPreview.getParent()).getWidth();
            } else {
                this.maxHeightOfImage = ((View) this.imgPreview.getParent()).getHeight();
            }
        }
        return this.maxHeightOfImage;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopAnalyzer();
    }

    private void stopAnalyzer() {
        if (mAnalyzer != null) {
            mAnalyzer.stop();
        }
    }
}