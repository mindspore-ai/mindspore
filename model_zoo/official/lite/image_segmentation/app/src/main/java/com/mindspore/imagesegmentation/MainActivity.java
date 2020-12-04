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
package com.mindspore.imagesegmentation;

import android.Manifest;
import android.content.Intent;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.mindspore.imagesegmentation.help.BitmapUtils;
import com.mindspore.imagesegmentation.help.ModelTrackingResult;
import com.mindspore.imagesegmentation.help.TrackingMobile;

import java.io.File;
import java.io.FileNotFoundException;
import java.lang.ref.WeakReference;

public class MainActivity extends AppCompatActivity implements OnBackgroundImageListener {
    private static final String TAG = "MainActivity";

    private static final int[] IMAGES = {R.drawable.img_001, R.drawable.img_002, R.drawable.img_003, R.drawable.img_004, R.drawable.img_005,
            R.drawable.img_006, R.drawable.img_007, R.drawable.img_008, R.drawable.add};

    private static final int REQUEST_PERMISSION = 0;
    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int RC_CHOOSE_PHOTO_FOR_BACKGROUND = 11;
    private static final int RC_CHOOSE_CAMERA = 2;

    private boolean isHasPermssion;

    private ImageView imgPreview;
    private TextView textOriginImage;
    public ProgressBar progressBar;
    private Uri imageUri;

    private Bitmap originBitmap, lastOriginBitmap;
    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    private boolean isLandScape;

    private TrackingMobile trackingMobile;
    private int selectedPosition;
    public boolean isRunningModel;
    private Handler mHandler;
    public ModelTrackingResult modelTrackingResult;
    private Bitmap processedImage;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.isLandScape = getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;
        setContentView(R.layout.activity_main);

        init();
        trackingMobile = new TrackingMobile(this);
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
            isHasPermssion = true;
        } else {
            requestPermissions();
        }
    }

    private void init() {
        imgPreview = findViewById(R.id.img_origin);
        textOriginImage = findViewById(R.id.tv_image);
        progressBar = findViewById(R.id.progress);
        RecyclerView recyclerView = findViewById(R.id.recyclerview);

        GridLayoutManager gridLayoutManager = new GridLayoutManager(this, 3);
        recyclerView.setLayoutManager(gridLayoutManager);
        recyclerView.setAdapter(new StyleRecyclerViewAdapter(this, IMAGES, this));
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

    public void onClickPhoto(View view) {
        if (isHasPermssion) {
            openGallay(RC_CHOOSE_PHOTO);
            textOriginImage.setVisibility(View.GONE);
        } else {
            requestPermissions();
        }
    }

    public void onClickCamera(View view) {
        if (isHasPermssion) {
            openCamera();
            textOriginImage.setVisibility(View.GONE);
        } else {
            requestPermissions();
        }
    }

    public void onClickRecovery(View view) {
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
        } else {
            Toast.makeText(this, "Please select an original picture first", Toast.LENGTH_SHORT).show();
        }
    }

    public void onClickSave(View view) {
        if (this.processedImage == null) {
            Log.e(TAG, "null processed image");
            Toast.makeText(this.getApplicationContext(), R.string.no_pic_neededSave, Toast.LENGTH_SHORT).show();
        } else {
            BitmapUtils.saveToAlbum(getApplicationContext(), this.processedImage);
            Toast.makeText(this.getApplicationContext(), R.string.save_success, Toast.LENGTH_SHORT).show();
        }
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
            } else if (RC_CHOOSE_PHOTO_FOR_BACKGROUND == requestCode) {
                if (null != data && null != data.getData()) {
                    showCustomBack(data.getData());
                } else {
                    finish();
                }
            } else if (RC_CHOOSE_CAMERA == requestCode) {
                if (null != data && null != data.getData()) {
                    this.imageUri = data.getData();
                    showOriginImage();
                }
                showOriginCamera();
            }
        }
    }

    private void showOriginImage() {
        Pair<Integer, Integer> targetedSize = this.getTargetSize();
        int targetWidth = targetedSize.first;
        int maxHeight = targetedSize.second;
        originBitmap = BitmapUtils.loadFromPath(this, imageUri, targetWidth, maxHeight);
        // Determine how much to scale down the image.
        Log.i(TAG, "resized image size width:" + originBitmap.getWidth() + ",height: " + originBitmap.getHeight());
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
        }
    }

    private void showOriginCamera() {
        try {
            originBitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        // Determine how much to scale down the image.
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
        }
    }

    private BitmapDrawable customBack;

    private void showCustomBack(Uri imageUri) {
        Pair<Integer, Integer> targetedSize = this.getTargetSize();
        int targetWidth = targetedSize.first;
        int maxHeight = targetedSize.second;
        Bitmap bitmap = BitmapUtils.loadFromPath(this, imageUri, targetWidth, maxHeight);
        customBack = new BitmapDrawable(getResources(), bitmap);
        changeBackground(false);
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

    private void startRunningModel(boolean isDemo) {
        if (!isRunningModel) {
            if (originBitmap == null) {
                Toast.makeText(this, "Please select an original picture first", Toast.LENGTH_SHORT).show();
                return;
            }
             progressBar.setVisibility(View.VISIBLE);
            new Thread(() -> {
                isRunningModel = true;
                modelTrackingResult = trackingMobile.execut(originBitmap);
                if (modelTrackingResult != null) {
                    isRunningModel = false;
                    lastOriginBitmap = originBitmap;
                }
                Looper.prepare();
                mHandler = new MyHandler(MainActivity.this, isDemo);
                mHandler.sendEmptyMessage(1);
                Looper.loop();
            }).start();
        } else {
            Toast.makeText(this, "Previous Model still running", Toast.LENGTH_SHORT).show();
        }
    }


    private static class MyHandler extends Handler {
        private final WeakReference<MainActivity> weakReference;
        private boolean isDemo;

        public MyHandler(MainActivity activity, boolean demo) {
            weakReference = new WeakReference<>(activity);
            this.isDemo = demo;
        }

        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            Log.i(TAG, "handleMessage: load");
            MainActivity activity = weakReference.get();
            if (msg.what == 1) {
                if (null != activity && null != activity.modelTrackingResult) {
                    activity.changeBackground(isDemo);
                }
            }
        }
    }

    public void changeBackground(boolean isDemo) {
        runOnUiThread(() -> {
            if(null != modelTrackingResult && null !=modelTrackingResult.getBitmapMaskOnly()){
                if (progressBar.getVisibility() == View.VISIBLE) {
                    progressBar.setVisibility(View.INVISIBLE);
                }
                Bitmap foreground = modelTrackingResult.getBitmapMaskOnly();
                Matrix matrix = new Matrix();
                matrix.setScale(0.7f, 0.7f);
                foreground = Bitmap.createBitmap(foreground, 0, 0, foreground.getWidth(), foreground.getHeight(), matrix, false);

                MainActivity.this.imgPreview.setImageBitmap(foreground);
                MainActivity.this.imgPreview.setDrawingCacheEnabled(true);
                MainActivity.this.imgPreview.setBackground(isDemo ? getDrawable(IMAGES[selectedPosition]) : customBack);
                MainActivity.this.imgPreview.setImageBitmap(foreground);
                MainActivity. this.processedImage = Bitmap.createBitmap( MainActivity.this.imgPreview.getDrawingCache());
                MainActivity.this.imgPreview.setDrawingCacheEnabled(false);
            }else{
                Toast.makeText(this, "Please select an original picture first", Toast.LENGTH_SHORT).show();
            }

        });

    }

    @Override
    public void onBackImageSelected(int position) {
        selectedPosition = position;
        if (lastOriginBitmap == originBitmap) {
            changeBackground(true);
        } else {
            startRunningModel(true);
        }
    }

    @Override
    public void onImageAdd(View view) {
        if (originBitmap == null) {
            Toast.makeText(this, "Please select an original picture first", Toast.LENGTH_SHORT).show();
            return;
        }
        if (lastOriginBitmap == originBitmap) {
            openGallay(RC_CHOOSE_PHOTO_FOR_BACKGROUND);
        } else {
            startRunningModel(false);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        trackingMobile.free();
        if (mHandler != null) {
            mHandler.removeCallbacksAndMessages(null);
        }
        BitmapUtils.recycleBitmap(this.originBitmap, this.lastOriginBitmap, this.processedImage);

    }

}