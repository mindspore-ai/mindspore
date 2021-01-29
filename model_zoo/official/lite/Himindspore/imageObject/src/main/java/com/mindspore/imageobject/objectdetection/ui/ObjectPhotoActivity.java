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

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
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

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.common.utils.Utils;
import com.mindspore.customview.dialog.NoticeDialog;
import com.mindspore.imageobject.R;
import com.mindspore.imageobject.objectdetection.bean.RecognitionObjectBean;
import com.mindspore.imageobject.objectdetection.help.ObjectTrackingMobile;
import com.mindspore.imageobject.util.BitmapUtils;
import com.mindspore.imageobject.util.DisplayUtil;

import static com.mindspore.imageobject.objectdetection.bean.RecognitionObjectBean.getRecognitionList;

@Route(path = "/imageobject/ObjectPhotoActivity")
public class ObjectPhotoActivity extends AppCompatActivity {

    private static final String TAG = "ObjectPhotoActivity";
    private static final int[] COLORS = {R.color.white, R.color.text_blue, R.color.text_yellow, R.color.text_orange, R.color.text_green};

    private static final int RC_CHOOSE_PHOTO = 1;
    private ImageView preview;
    private ObjectTrackingMobile trackingMobile;
    private List<RecognitionObjectBean> recognitionObjectBeanList;

    private Bitmap originBitmap;
    private Uri imageUri;
    private NoticeDialog noticeDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "onCreate ObjectPhotoActivity info");
        setContentView(R.layout.activity_object_photo);
        init();
        preview = findViewById(R.id.img_photo);
        openGallay();
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
            this.imageUri = data.getData();
            showOriginImage();
        } else {
            Toast.makeText(this, R.string.image_invalid, Toast.LENGTH_LONG).show();
            finish();
        }
    }

    private void showOriginImage() {
        File file = BitmapUtils.getFileFromMediaUri(this, imageUri);
        Bitmap photoBmp = BitmapUtils.getBitmapFormUri(this, Uri.fromFile(file));
        int degree = BitmapUtils.getBitmapDegree(file.getAbsolutePath());
        originBitmap = BitmapUtils.rotateBitmapByDegree(photoBmp, degree).copy(Bitmap.Config.ARGB_8888, true);
        if (originBitmap != null) {
            initMindspore(originBitmap);
            preview.setImageBitmap(originBitmap);
        } else {
            Toast.makeText(this, R.string.image_invalid, Toast.LENGTH_LONG).show();
        }
    }

    private void initMindspore(Bitmap bitmap) {
        try {
            trackingMobile = new ObjectTrackingMobile(this);
        } catch (FileNotFoundException e) {
            Log.e(TAG, Log.getStackTraceString(e));
            e.printStackTrace();
        }
        boolean ret = trackingMobile.loadModelFromBuf(getAssets());
        if (!ret) {
            Log.e(TAG, "Load model error.");
            return;
        }
        // run net.
        long startTime = System.currentTimeMillis();
        String result = trackingMobile.MindSpore_runnet(bitmap);
        long endTime = System.currentTimeMillis();

        Log.d(TAG, "RUNNET CONSUMING：" + (endTime - startTime) + "ms");
        Log.d(TAG, "result：" + result);

        recognitionObjectBeanList = getRecognitionList(result);

        if (recognitionObjectBeanList != null && recognitionObjectBeanList.size() > 0) {
            drawRect(bitmap);
        } else {
            Toast.makeText(this, R.string.train_invalid, Toast.LENGTH_LONG).show();
        }
    }

    private void drawRect(Bitmap bitmap) {
        Canvas canvas = new Canvas(bitmap);
        Paint mPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        //Draw only outline (stroke)
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeWidth(DisplayUtil.dip2px(this, 2));

        Paint mPaintText = new Paint(Paint.ANTI_ALIAS_FLAG);
        mPaintText.setTextSize(DisplayUtil.sp2px(this, 16));
        mPaintText.setStyle(Paint.Style.FILL);

        for (int i = 0; i < recognitionObjectBeanList.size(); i++) {
            RecognitionObjectBean objectBean = recognitionObjectBeanList.get(i);
            StringBuilder sb = new StringBuilder();
            sb.append(objectBean.getRectID()).append("-").append(objectBean.getObjectName()).append("-").append(String.format("%.2f", (100 * objectBean.getScore())) + "%");

            int paintColor = getResources().getColor(COLORS[i % COLORS.length]);
            mPaint.setColor(paintColor);
            mPaintText.setColor(paintColor);

            RectF rectF = new RectF(objectBean.getLeft(), objectBean.getTop(), objectBean.getRight(), objectBean.getBottom());
            canvas.drawRect(rectF, mPaint);
            if (rectF.top < DisplayUtil.dip2px(this, 20)){
                canvas.drawText(sb.toString(), rectF.left + DisplayUtil.dip2px(this, 5), rectF.top + DisplayUtil.dip2px(this, 20), mPaintText);
            }else{
                canvas.drawText(sb.toString(), rectF.left, rectF.top - DisplayUtil.dip2px(this, 10), mPaintText);
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (trackingMobile != null) {
            trackingMobile.unloadModel();
        }
        BitmapUtils.recycleBitmap(originBitmap);
    }

    private void init() {
        Toolbar mToolbar = findViewById(R.id.activity_photo_toolbar);
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
            Utils.openBrowser(this, MSLinkUtils.HELP_PHOTO_DETECTION);
        }
        return super.onOptionsItemSelected(item);
    }

    private void showHelpDialog() {
        noticeDialog = new NoticeDialog(this);
        noticeDialog.setTitleString(getString(R.string.explain_title));
        noticeDialog.setContentString(getString(R.string.explain_photo_detection));
        noticeDialog.setYesOnclickListener(() -> {
            noticeDialog.dismiss();
        });
        noticeDialog.show();
    }
}