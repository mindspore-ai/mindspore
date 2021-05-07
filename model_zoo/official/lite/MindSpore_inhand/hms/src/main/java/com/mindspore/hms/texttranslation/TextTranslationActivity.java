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
package com.mindspore.hms.texttranslation;

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
import android.widget.EditText;
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
import com.huawei.hms.mlsdk.common.MLApplication;
import com.huawei.hms.mlsdk.common.MLException;
import com.huawei.hms.mlsdk.common.MLFrame;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentationAnalyzer;
import com.huawei.hms.mlsdk.text.MLLocalTextSetting;
import com.huawei.hms.mlsdk.text.MLText;
import com.huawei.hms.mlsdk.text.MLTextAnalyzer;
import com.huawei.hms.mlsdk.translate.MLTranslatorFactory;
import com.huawei.hms.mlsdk.translate.cloud.MLRemoteTranslateSetting;
import com.huawei.hms.mlsdk.translate.cloud.MLRemoteTranslator;
import com.mindspore.common.sp.Preferences;
import com.mindspore.common.utils.StringUtils;
import com.mindspore.hms.BitmapUtils;
import com.mindspore.hms.R;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;

@Route(path = "/hms/TextTranslationActivity")
public class TextTranslationActivity extends AppCompatActivity {

    private static final String TAG = TextTranslationActivity.class.getSimpleName();

    private EditText mEditText;
    private TextView mTextView;
    private MLRemoteTranslator mlRemoteTranslator;


    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int RC_CHOOSE_CAMERA = 2;

    private boolean isPreViewShow = false;

    private ImageView imgPreview, mImageView;
    private Uri imageUri;

    private Bitmap originBitmap;
    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    private boolean isLandScape;
    private MLImageSegmentationAnalyzer analyzer;
    private MLFrame frame;
    private final int analyzerType = -1;
    private Bitmap bitmapFore;
    private MLTextAnalyzer mAnalyzer;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text_translation);
        MLApplication.getInstance().setApiKey(Preferences.API_KEY);
        init();
    }

    private void init() {
        Toolbar mToolbar = findViewById(R.id.text_online_activity_toolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
        mEditText = findViewById(R.id.text_edit);
        mEditText.setMovementMethod(ScrollingMovementMethod.getInstance());
        mTextView = findViewById(R.id.text_view);
        imgPreview = findViewById(R.id.img_origin);
        mTextView.setMovementMethod(ScrollingMovementMethod.getInstance());

        MLLocalTextSetting setting = new MLLocalTextSetting.Factory()
                .setOCRMode(MLLocalTextSetting.OCR_DETECT_MODE)
                .setLanguage("zh")
                .create();
        mAnalyzer = MLAnalyzerFactory.getInstance().getLocalTextAnalyzer(setting);
    }


    private void remoteTranslator() {
        String sourceText = mEditText.getText().toString();

        MLRemoteTranslateSetting setting = new MLRemoteTranslateSetting
                .Factory()
                .setTargetLangCode(StringUtils.isChinese(sourceText) ? "en" : "zh")
                .create();


        mlRemoteTranslator = MLTranslatorFactory.getInstance().getRemoteTranslator(setting);

        Task<String> task = mlRemoteTranslator.asyncTranslate(sourceText);
        task.addOnSuccessListener(new OnSuccessListener<String>() {
            @Override
            public void onSuccess(String text) {
                mTextView.setText(text);
            }

        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(Exception e) {

                TextTranslationActivity.this.displayFailure(e);
            }
        });
    }


    private void displayFailure(Exception exception) {
        String error = "Failure. ";
        if (exception instanceof MLException) {
            MLException mlException = (MLException) exception;
            error += "error code: " + mlException.getErrCode() + "\n" + "error message: " + mlException.getMessage();
        } else {
            error += exception.getMessage();
        }
        this.mTextView.setText(error);
    }

    public void onClickTranslation(View view) {
        this.remoteTranslator();

    }

    public void onClickTextCopy(View view) {
        String s = mTextView.getText().toString();
        if (!s.equals("")) {
            ClipboardManager cmb = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            cmb.setText(mTextView.getText());
            Toast.makeText(this, R.string.text_copied_successfully, Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, R.string.text_null, Toast.LENGTH_SHORT).show();
        }
    }

    public void onClickPhoto(View view) {
        openGallay(RC_CHOOSE_PHOTO);
        mEditText.setVisibility(View.GONE);
        imgPreview.setVisibility(View.VISIBLE);
    }

    public void onClickCamera(View view) {
        openCamera();
        mEditText.setVisibility(View.GONE);
        imgPreview.setVisibility(View.VISIBLE);
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

    private void showTextRecognition() {
        MLFrame frame = MLFrame.fromBitmap(originBitmap);

        Task<MLText> task = mAnalyzer.asyncAnalyseFrame(frame);
        task.addOnSuccessListener(new OnSuccessListener<MLText>() {
            @Override
            public void onSuccess(MLText text) {
                TextTranslationActivity.this.displaySuccess(text);
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
//        remoteTranslator(result);
//        result.trim();
//        string.replaceAll("[^0-9a-zA-Z]","");
        MLRemoteTranslateSetting setting = new MLRemoteTranslateSetting
                .Factory()
                .setTargetLangCode(StringUtils.isChinese(result) ? "en" : "zh")
                .create();


        mlRemoteTranslator = MLTranslatorFactory.getInstance().getRemoteTranslator(setting);

        Task<String> task = mlRemoteTranslator.asyncTranslate(result);
        task.addOnSuccessListener(new OnSuccessListener<String>() {
            @Override
            public void onSuccess(String text) {
                mTextView.setText(text);
            }

        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(Exception e) {

            }
        });

//        this.mTextView.setText(result);
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
        if (mlRemoteTranslator != null) {
            mlRemoteTranslator.stop();
        }
    }
}