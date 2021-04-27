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
package com.mindspore.hms.textrecognition;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.ClipboardManager;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.bumptech.glide.Glide;
import com.huawei.hmf.tasks.OnFailureListener;
import com.huawei.hmf.tasks.OnSuccessListener;
import com.huawei.hmf.tasks.Task;
import com.huawei.hms.mlsdk.MLAnalyzerFactory;
import com.huawei.hms.mlsdk.common.MLFrame;
import com.huawei.hms.mlsdk.text.MLLocalTextSetting;
import com.huawei.hms.mlsdk.text.MLText;
import com.huawei.hms.mlsdk.text.MLTextAnalyzer;
import com.mindspore.hms.BitmapUtils;
import com.mindspore.hms.R;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

@Route(path = "/hms/TextRecognitionActivity")
public class TextRecognitionActivity extends AppCompatActivity {

    private static final String TAG = "TextRecognitionActivity";

    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int RC_CHOOSE_CAMERA = 2;

    private boolean isPreViewShow = false;

    private ImageView imgPreview;
    private TextView textOriginImage;
    private Uri imageUri;

    private Bitmap originBitmap;

    private TextView mTextView;
    private Button mBtnCopy;
    private MLTextAnalyzer analyzer;
    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    private boolean isLandScape;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text_recognition);
        init();
    }

    private void init() {
        mTextView = findViewById(R.id.text_text);
        mBtnCopy = findViewById(R.id.btn_copy);
        mTextView.setMovementMethod(ScrollingMovementMethod.getInstance());
        MLLocalTextSetting setting = new MLLocalTextSetting.Factory()
                .setOCRMode(MLLocalTextSetting.OCR_DETECT_MODE)
                .setLanguage("zh")
                .create();
        analyzer = MLAnalyzerFactory.getInstance().getLocalTextAnalyzer(setting);
        imgPreview = findViewById(R.id.img_origin);
        textOriginImage = findViewById(R.id.tv_image);
        Toolbar mToolbar = findViewById(R.id.segmentation_toolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());

    }

    public void onClickPhoto(View view) {
        openGallay(RC_CHOOSE_PHOTO);
        textOriginImage.setVisibility(View.GONE);
    }

    public void onClickCopy(View view) {
        String s = mTextView.getText().toString();
        if (!s.equals("")) {
            ClipboardManager cmb = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            cmb.setText(mTextView.getText());
            Toast.makeText(this, R.string.text_copied_successfully, Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, R.string.text_null, Toast.LENGTH_SHORT).show();
        }
    }

    public void onClickCamera(View view) {
        openCamera();
        textOriginImage.setVisibility(View.GONE);
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
            }
        } else {
            textOriginImage.setVisibility(!isPreViewShow ? View.VISIBLE : View.GONE);
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
            showTextRecognition();
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
            showTextRecognition();
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

    private void showTextRecognition() {
        MLFrame frame = MLFrame.fromBitmap(originBitmap);

        Task<MLText> task = analyzer.asyncAnalyseFrame(frame);
        task.addOnSuccessListener(new OnSuccessListener<MLText>() {
            @Override
            public void onSuccess(MLText text) {
                TextRecognitionActivity.this.displaySuccess(text);
            }
        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(Exception e) {

            }
        });
    }

    private void displaySuccess(MLText mlText) {
        String result = "";
        List<MLText.Block> blocks = mlText.getBlocks();
        for (MLText.Block block : blocks) {
            for (MLText.TextLine line : block.getContents()) {
                result += line.getStringValue() + "\n";
            }
        }
        this.mTextView.setText(result);
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (this.analyzer == null) {
            return;
        }
        try {
            this.analyzer.stop();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}