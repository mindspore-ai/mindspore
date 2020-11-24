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
package com.mindspore.styletransferdemo;

import android.Manifest;
import android.content.Intent;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.bumptech.glide.Glide;

public class MainActivity extends AppCompatActivity implements View.OnClickListener, StyleFragment.OnListFragmentInteractionListener {

    private static final String TAG = "MainActivity";

    private static final int REQUEST_PERMISSION = 1;
    private static final int RC_CHOOSE_PHOTO = 2;

    private StyleTransferModelExecutor transferModelExecutor;

    private boolean isHasPermssion;
    private boolean isRunningModel;

    private ImageView imgOrigin, imgStyle, imgResult;
    private ProgressBar progressResult;
    private Uri imageUri;

    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    private boolean isLandScape;

    private Bitmap originBitmap, styleBitmap;
    private StyleFragment styleFragment;
    private String selectedStyle;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        this.isLandScape = getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;
        requestPermissions();
        init();
    }

    private void init() {
        imgOrigin = findViewById(R.id.img_origin);
        imgStyle = findViewById(R.id.img_style);
        imgResult = findViewById(R.id.img_result);
        progressResult = findViewById(R.id.progress_circular);

        imgOrigin.setOnClickListener(this);
        imgStyle.setOnClickListener(this);
        imgResult.setOnClickListener(this);

        styleFragment = StyleFragment.newInstance();
        transferModelExecutor = new StyleTransferModelExecutor(this, false);
    }

    private void requestPermissions() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE,
                        Manifest.permission.READ_PHONE_STATE, Manifest.permission.CAMERA}, REQUEST_PERMISSION);
    }

    /**
     * Authority application result callback
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (REQUEST_PERMISSION == requestCode) {
            isHasPermssion = true;
        }
    }


    @Override
    public void onClick(View view) {
        if (view.getId() == R.id.img_origin) {
            if (isHasPermssion) {
                openGallay();
            } else {
                requestPermissions();
            }
        } else if (view.getId() == R.id.img_style) {
            if (!isRunningModel) {
                styleFragment.show(getSupportFragmentManager(), TAG);
            }
        }
    }

    private void openGallay() {
        Intent intentToPickPic = new Intent(Intent.ACTION_PICK, null);
        intentToPickPic.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intentToPickPic, RC_CHOOSE_PHOTO);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (RC_CHOOSE_PHOTO == requestCode && null != data && null != data.getData()) {
            if (data != null) {
                this.imageUri = data.getData();
                showOriginImage();
            }
        } else {
            finish();
        }
    }

    private void showOriginImage() {
        Pair<Integer, Integer> targetedSize = this.getTargetSize();
        int targetWidth = targetedSize.first;
        int maxHeight = targetedSize.second;
        originBitmap = BitmapUtils.loadFromPath(MainActivity.this, imageUri, targetWidth, maxHeight);
        // Determine how much to scale down the image.
        Log.i(TAG, "resized image size width:" + originBitmap.getWidth() + ",height: " + originBitmap.getHeight());

        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgOrigin);
        }
    }


    @Override
    public void onListFragmentInteraction(String item) {
        this.selectedStyle = item;
        styleFragment.dismiss();
        startRunningModel();
    }

    private void startRunningModel() {
        if (!isRunningModel && !TextUtils.isEmpty(selectedStyle)) {
            styleBitmap = ImageUtils.loadBitmapFromResources(this, getUriFromAssetThumb(selectedStyle));
            Glide.with(this)
                    .load(styleBitmap)
                    .into(imgStyle);

            if (originBitmap == null) {
                Toast.makeText(this, "Please select an original picture first", Toast.LENGTH_SHORT).show();
                return;
            }
            progressResult.setVisibility(View.VISIBLE);
            isRunningModel = true;
            ModelExecutionResult result = transferModelExecutor.execute(originBitmap, styleBitmap);
            Glide.with(this).load(result.getStyledImage()).into(imgResult);
            progressResult.setVisibility(View.GONE);
            isRunningModel = false;
        } else {
            Toast.makeText(this, "Previous Model still running", Toast.LENGTH_SHORT).show();
        }
    }

    private String getUriFromAssetThumb(String thumb) {
        return "thumbnails/" + thumb;
    }

    // Returns max width of image.
    private Integer getMaxWidthOfImage() {
        if (this.maxWidthOfImage == null) {
            if (this.isLandScape) {
                this.maxWidthOfImage = ((View) this.imgOrigin.getParent()).getHeight();
            } else {
                this.maxWidthOfImage = ((View) this.imgOrigin.getParent()).getWidth();
            }
        }
        return this.maxWidthOfImage;
    }

    // Returns max height of image.
    private Integer getMaxHeightOfImage() {
        if (this.maxHeightOfImage == null) {
            if (this.isLandScape) {
                this.maxHeightOfImage = ((View) this.imgOrigin.getParent()).getWidth();
            } else {
                this.maxHeightOfImage = ((View) this.imgOrigin.getParent()).getHeight();
            }
        }
        return this.maxHeightOfImage;
    }

    // Gets the targeted size(width / height).
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


}