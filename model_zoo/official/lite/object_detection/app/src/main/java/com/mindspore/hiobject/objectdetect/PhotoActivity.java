package com.mindspore.hiobject.objectdetect;

import android.content.Intent;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.hiobject.R;
import com.mindspore.hiobject.help.BitmapUtils;
import com.mindspore.hiobject.help.DisplayUtil;
import com.mindspore.hiobject.help.RecognitionObjectBean;
import com.mindspore.hiobject.help.TrackingMobile;

import java.io.FileNotFoundException;
import java.util.List;

import static com.mindspore.hiobject.help.RecognitionObjectBean.getRecognitionList;

public class PhotoActivity extends AppCompatActivity {

    private static final String TAG = "PhotoActivity";
    private static final int[] COLORS = {R.color.white, R.color.text_blue, R.color.text_yellow, R.color.text_orange, R.color.text_green};

    private static final int RC_CHOOSE_PHOTO = 1;

    private ImageView preview;
    private TrackingMobile trackingMobile;
    private List<RecognitionObjectBean> recognitionObjectBeanList;


    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    boolean isLandScape;
    private Bitmap originBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_photo);
        preview = findViewById(R.id.img_photo);
        this.isLandScape = getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;
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
            if (data != null) {
                showOriginImage(data.getData());
            }
        } else {
            Toast.makeText(this, R.string.image_invalid, Toast.LENGTH_LONG).show();
            finish();
        }
    }

    private void showOriginImage(Uri imageUri) {
        Pair<Integer, Integer> targetedSize = this.getTargetSize();
        int targetWidth = targetedSize.first;
        int maxHeight = targetedSize.second;
        originBitmap = BitmapUtils.loadFromPath(this, imageUri, targetWidth, maxHeight).copy(Bitmap.Config.ARGB_8888, true);
        // Determine how much to scale down the image.
        Log.i(TAG, "resized image size width:" + originBitmap.getWidth() + ",height: " + originBitmap.getHeight());
        if (originBitmap != null) {
            initMindspore(originBitmap);
            preview.setImageBitmap(originBitmap);
        } else {
            Toast.makeText(this, R.string.image_invalid, Toast.LENGTH_LONG).show();
        }
    }

    private void initMindspore(Bitmap bitmap) {
        try {
            trackingMobile = new TrackingMobile(this);
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
        mPaint.setTextSize(DisplayUtil.sp2px(this, 16));
        //Draw only outline (stroke)
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeWidth(DisplayUtil.dip2px(this, 2));

        for (int i = 0; i < recognitionObjectBeanList.size(); i++) {
            RecognitionObjectBean objectBean = recognitionObjectBeanList.get(i);
            StringBuilder sb = new StringBuilder();
            sb.append(objectBean.getRectID()).append("_").append(objectBean.getObjectName()).append("_").append(String.format("%.2f", (100 * objectBean.getScore())) + "%");

            int paintColor = getResources().getColor(COLORS[i % COLORS.length]);
            mPaint.setColor(paintColor);

            RectF rectF = new RectF(objectBean.getLeft(), objectBean.getTop(), objectBean.getRight(), objectBean.getBottom());
            canvas.drawRect(rectF, mPaint);
            canvas.drawText(sb.toString(), objectBean.getLeft(), objectBean.getTop() - 10, mPaint);
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


    // Returns max width of image.
    private Integer getMaxWidthOfImage() {
        if (this.maxWidthOfImage == null) {
            if (this.isLandScape) {
                this.maxWidthOfImage = ((View) this.preview.getParent()).getHeight();
            } else {
                this.maxWidthOfImage = ((View) this.preview.getParent()).getWidth();
            }
        }
        return this.maxWidthOfImage;
    }

    // Returns max height of image.
    private Integer getMaxHeightOfImage() {
        if (this.maxHeightOfImage == null) {
            if (this.isLandScape) {
                this.maxHeightOfImage = ((View) this.preview.getParent()).getWidth();
            } else {
                this.maxHeightOfImage = ((View) this.preview.getParent()).getHeight();
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