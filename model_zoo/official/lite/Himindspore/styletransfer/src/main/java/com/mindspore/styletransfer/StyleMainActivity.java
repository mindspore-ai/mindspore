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
package com.mindspore.styletransfer;

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
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.bumptech.glide.Glide;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Route(path = "/styletransfer/StyleMainActivity")
public class StyleMainActivity extends AppCompatActivity implements View.OnClickListener, OnListFragmentInteractionListener {

    private static final String TAG = "StyleMainActivity";

    private static final int RC_CHOOSE_PHOTO = 1;

    private StyleTransferModelExecutor transferModelExecutor;

    private boolean isRunningModel;

    private ImageView imgOrigin;
    private Button btnImage;
    private Uri imageUri;

    private RecyclerView recyclerView;

    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    private boolean isLandScape;

    private Bitmap originBitmap, styleBitmap;
    private String selectedStyle;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_style);
        this.isLandScape = getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;
        init();
    }

    private void init() {
        imgOrigin = findViewById(R.id.img_origin);
        btnImage = findViewById(R.id.btn_image);
        imgOrigin.setOnClickListener(this);
        btnImage.setOnClickListener(this);

        recyclerView = findViewById(R.id.recyclerview);
        List<String> styles = new ArrayList<>();
        try {
            styles.addAll(Arrays.asList(getAssets().list("thumbnails")));
        } catch (IOException e) {
            e.printStackTrace();
        }

        GridLayoutManager gridLayoutManager = new GridLayoutManager(this, 3);
        recyclerView.setLayoutManager(gridLayoutManager);
        recyclerView.setAdapter(new StyleRecyclerViewAdapter(this, styles, this));

        transferModelExecutor = new StyleTransferModelExecutor(this, false);
    }


    @Override
    public void onClick(View view) {
        if (view.getId() == R.id.img_origin || view.getId() == R.id.btn_image) {
            btnImage.setVisibility(View.GONE);
            openGallay();
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
        originBitmap = BitmapUtils.loadFromPath(StyleMainActivity.this, imageUri, targetWidth, maxHeight);
        // Determine how much to scale down the image.
        Log.i(TAG, "resized image size width:" + originBitmap.getWidth() + ",height: " + originBitmap.getHeight());

        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgOrigin);
        }
    }


    @Override
    public void onListFragmentInteraction(String item) {
        this.selectedStyle = item;
        startRunningModel();
    }

    private void startRunningModel() {
        if (!isRunningModel && !TextUtils.isEmpty(selectedStyle)) {
            styleBitmap = ImageUtils.loadBitmapFromResources(this, getUriFromAssetThumb(selectedStyle));
            if (originBitmap == null) {
                Toast.makeText(this, "Please select an original picture first", Toast.LENGTH_SHORT).show();
                return;
            }
            isRunningModel = true;
            ModelExecutionResult result = transferModelExecutor.execute(originBitmap, styleBitmap);
            Glide.with(this).load(result.getStyledImage()).into(imgOrigin);
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