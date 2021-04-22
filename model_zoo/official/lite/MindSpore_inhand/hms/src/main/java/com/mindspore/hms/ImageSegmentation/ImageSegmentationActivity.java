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
package com.mindspore.hms.ImageSegmentation;

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
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.alibaba.android.arouter.launcher.ARouter;
import com.bumptech.glide.Glide;
import com.huawei.hmf.tasks.Task;
import com.huawei.hms.mlsdk.MLAnalyzerFactory;
import com.huawei.hms.mlsdk.common.MLFrame;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentation;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentationAnalyzer;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentationScene;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentationSetting;
import com.mindspore.hms.BitmapUtils;
import com.mindspore.hms.R;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

@Route(path = "/hms/ImageSegmentationActivity")
public class ImageSegmentationActivity extends AppCompatActivity {

    private static final String TAG = ImageSegmentationActivity.class.getSimpleName();
    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int RC_CHOOSE_CAMERA = 2;

    private ImageView imgPreview, mImageView;
    private Uri imageUri;

    private Bitmap originBitmap;
    private MLImageSegmentationAnalyzer analyzer;
    private Bitmap bitmapFore;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_segmentation);
        createAnalyzer();
        init();
    }

    private void init() {
        imgPreview = findViewById(R.id.img_origin);
        mImageView = findViewById(R.id.image_result);

        Toolbar mToolbar = findViewById(R.id.ImageSegmentation_activity_toolbar);
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
    }


    private void createAnalyzer() {
        MLImageSegmentationSetting setting = new MLImageSegmentationSetting.Factory()
                .setExact(false)
                .setAnalyzerType(MLImageSegmentationSetting.BODY_SEG)
                .setScene(MLImageSegmentationScene.ALL)
                .create();
        analyzer = MLAnalyzerFactory.getInstance().getImageSegmentationAnalyzer(setting);
    }

    public void onClickPhoto(View view) {
        openGallay(RC_CHOOSE_PHOTO);
    }

    public void onClickCamera(View view) {
        openCamera();
    }


    public void onClickRealTime(View view) {
        ARouter.getInstance().build("/hms/ImageSegmentationLiveAnalyseActivity").navigation();
    }

    private void openGallay(int request) {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, request);
    }

    public void onClickSave(View view) {
        if (this.bitmapFore == null) {
            Log.e(TAG, "null processed image");
            Toast.makeText(this.getApplicationContext(), R.string.no_pic_neededSave, Toast.LENGTH_SHORT).show();
        } else {
            BitmapUtils.saveToAlbum(getApplicationContext(), this.bitmapFore);
            Toast.makeText(this.getApplicationContext(), R.string.save_success, Toast.LENGTH_SHORT).show();
        }
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
            }
        }
    }

    private void showOriginImage() {
        try {
            originBitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
            showSegmentationImage();
        } else {
        }
    }

    private void showSegmentationImage() {
        MLFrame frame = new MLFrame.Creator().setBitmap(originBitmap).create();
        Task<MLImageSegmentation> task = analyzer.asyncAnalyseFrame(frame);
        task.addOnSuccessListener(segmentation -> {
            // Processing logic for recognition success.
            if (segmentation != null) {
                displaySuccess(segmentation);
            }
        }).addOnFailureListener(e -> {
            displayFailure(e.toString());
        });
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
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
            showSegmentationImage();
        }
    }

    private Pair<Integer, Integer> getTargetSize() {
        return new Pair<>(((View) this.imgPreview.getParent()).getWidth(), ((View) this.imgPreview.getParent()).getHeight());
    }


    private void displaySuccess(MLImageSegmentation imageSegmentationResult) {
        // Draw the portrait with a transparent background.
        bitmapFore = imageSegmentationResult.getForeground();
        if (bitmapFore != null) {
            this.mImageView.setImageBitmap(bitmapFore);
        } else {
            this.displayFailure("bitmapFore is null.");
        }
    }

    private void displayFailure(String str) {
        Log.e(TAG, str);
        Toast.makeText(this, R.string.segmentation_fail, Toast.LENGTH_SHORT).show();
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopAnalyzer();
    }

    private void stopAnalyzer() {
        if (analyzer != null) {
            try {
                analyzer.stop();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}