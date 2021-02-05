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
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Pair;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.bumptech.glide.Glide;
import com.mindspore.common.base.grid.MSGridSpacingItemDecoration;
import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.common.utils.Utils;
import com.mindspore.customview.dialog.NoticeDialog;

import java.io.File;
import java.io.FileNotFoundException;

@Route(path = "/styletransfer/StyleMainActivity")
public class StyleMainActivity extends AppCompatActivity implements OnBackgroundImageListener {

    private static final String TAG = "StyleMainActivity";

    private static final int[] IMAGES = {R.drawable.style0, R.drawable.style1, R.drawable.style2, R.drawable.style3, R.drawable.style4,
            R.drawable.style5, R.drawable.style6, R.drawable.style7, R.drawable.style8, R.drawable.style9,
            R.drawable.style10, R.drawable.style11, R.drawable.style12, R.drawable.style13, R.drawable.style14,
            R.drawable.style15, R.drawable.style16, R.drawable.style17, R.drawable.style18, R.drawable.icon_default};

    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int RC_CHOOSE_PHOTO_FOR_BACKGROUND = 11;
    private static final int RC_CHOOSE_CAMERA = 2;


    private boolean isPreViewShow = false;
    private StyleTransferModelExecutor transferModelExecutor;

    private boolean isRunningModel;

    private ImageView imgPreview;
    private Uri imageUri;
    private TextView textOriginImage;
    private ProgressBar progressBar;

    private RecyclerView recyclerView;

    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    private boolean isLandScape;

    private Bitmap originBitmap, styleBitmap, resultBitmap;
    private NoticeDialog noticeDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_style);
        this.isLandScape = getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;
        init();
    }

    private void init() {
        imgPreview = findViewById(R.id.img_origin);
        textOriginImage = findViewById(R.id.tv_image);
        progressBar = findViewById(R.id.progress);
        recyclerView = findViewById(R.id.recyclerview);

        recyclerView.setLayoutManager(new GridLayoutManager(this, 4));
        recyclerView.addItemDecoration(new MSGridSpacingItemDecoration(10));
        recyclerView.setAdapter(new StyleRecyclerViewAdapter(this, IMAGES, this));
        transferModelExecutor = new StyleTransferModelExecutor(this, false);

        Toolbar mToolbar = findViewById(R.id.style_transfer_toolbar);
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
            Utils.openBrowser(this, MSLinkUtils.HELP_STYLE_TRANSFER);
        }
        return super.onOptionsItemSelected(item);
    }


    private void showHelpDialog() {
        noticeDialog = new NoticeDialog(this);
        noticeDialog.setTitleString(getString(R.string.explain_title));
        noticeDialog.setContentString(getString(R.string.explain_style_transfer));
        noticeDialog.setYesOnclickListener(() -> {
            noticeDialog.dismiss();
        });
        noticeDialog.show();
    }

    public void onClickPhoto(View view) {
        openGallay(RC_CHOOSE_PHOTO);
        textOriginImage.setVisibility(View.GONE);
    }

    public void onClickCamera(View view) {
        openCamera();
        textOriginImage.setVisibility(View.GONE);
    }

    public void onClickRecovery(View view) {
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
            isPreViewShow = true;
        } else {
            Toast.makeText(this, R.string.toast_original, Toast.LENGTH_SHORT).show();
            isPreViewShow = false;
        }
    }

    public void onClickSave(View view) {
        if (this.resultBitmap == null) {
            Log.e(TAG, "null processed image");
            Toast.makeText(this.getApplicationContext(), R.string.no_pic_neededSave, Toast.LENGTH_SHORT).show();
        } else {
            ImageUtils.saveToAlbum(getApplicationContext(), this.resultBitmap);
            Toast.makeText(this.getApplicationContext(), R.string.save_success, Toast.LENGTH_SHORT).show();
        }
    }

    private void openGallay(int request) {
        Intent intentToPickPic = new Intent(Intent.ACTION_PICK, null);
        intentToPickPic.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intentToPickPic, request);
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
            } else if (RC_CHOOSE_PHOTO_FOR_BACKGROUND == requestCode) {
                if (null != data && null != data.getData()) {
                    showCustomBack(data.getData());
                } else {
                    finish();
                }
            } else if (RC_CHOOSE_CAMERA == requestCode) {
                showOriginCamera();
            }
        } else {
            textOriginImage.setVisibility(!isPreViewShow ? View.VISIBLE : View.GONE);
        }
    }

    private void showOriginImage() {
        Pair<Integer, Integer> targetedSize = this.getTargetSize();
        int targetWidth = targetedSize.first;
        int maxHeight = targetedSize.second;
        originBitmap = BitmapUtils.loadFromPath(this, imageUri, targetWidth, maxHeight);
        // Determine how much to scale down the image.
        Log.e(TAG, "resized image size width:" + originBitmap.getWidth() + ",height: " + originBitmap.getHeight());
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
            isPreViewShow = true;
        } else {
            isPreViewShow = false;
        }
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
        }else {
            isPreViewShow = false;
        }
    }

    private void showCustomBack(Uri imageUri) {
        Pair<Integer, Integer> targetedSize = this.getTargetSize();
        int targetWidth = targetedSize.first;
        int maxHeight = targetedSize.second;
        styleBitmap = BitmapUtils.loadFromPath(this, imageUri, targetWidth, maxHeight);
        startRunningModel(styleBitmap);
    }

    @Override
    public void onBackImageSelected(int position) {
        styleBitmap = BitmapFactory.decodeResource(getResources(), IMAGES[position]);
        startRunningModel(styleBitmap);
    }

    @Override
    public void onImageAdd(View view) {
        openGallay(RC_CHOOSE_PHOTO_FOR_BACKGROUND);
    }

    private void startRunningModel(Bitmap styleBitmap) {
        if (originBitmap == null) {
            Toast.makeText(this, R.string.toast_original, Toast.LENGTH_SHORT).show();
            return;
        }

        if (!isRunningModel) {
            isRunningModel = true;
            progressBar.setVisibility(View.VISIBLE);
            ModelExecutionResult result = transferModelExecutor.execute(originBitmap, styleBitmap);
            if (null != result && null != result.getStyledImage()) {
                resultBitmap = BitmapUtils.changeBitmapSize(result.getStyledImage(), originBitmap.getWidth(), originBitmap.getHeight());
                Glide.with(this).load(resultBitmap).override(resultBitmap.getWidth(), resultBitmap.getHeight()).into(imgPreview);
                isPreViewShow = true;
            } else {
                Toast.makeText(this, R.string.toast_execute_fail, Toast.LENGTH_SHORT).show();
                isPreViewShow = false;
            }
            isRunningModel = false;
            progressBar.setVisibility(View.INVISIBLE);
        } else {
            Toast.makeText(this, R.string.toast_model_run, Toast.LENGTH_SHORT).show();
        }
    }

    // Returns max width of image.
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

    // Returns max height of image.
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