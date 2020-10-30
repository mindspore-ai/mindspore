package com.mindspore.himindspore.objectdetection.ui;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;


import com.mindspore.himindspore.R;
import com.mindspore.himindspore.objectdetection.help.ObjectTrackingMobile;
import com.mindspore.himindspore.objectdetection.bean.RecognitionObjectBean;
import com.mindspore.himindspore.objectdetection.help.ImageDegreeHelper;
import com.mindspore.himindspore.utils.DisplayUtil;

import java.io.FileNotFoundException;
import java.util.List;

import static com.mindspore.himindspore.objectdetection.bean.RecognitionObjectBean.getRecognitionList;


public class PhotoActivity extends AppCompatActivity {

    private static final String TAG = "PhotoActivity";
    private static final int[] COLORS ={R.color.white,R.color.text_blue,R.color.text_yellow,R.color.text_orange,R.color.text_green};

    private ImageView imgPhoto;
    private ObjectTrackingMobile trackingMobile;
    private List<RecognitionObjectBean> recognitionObjectBeanList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_object_photo);

        imgPhoto = findViewById(R.id.img_photo);

        Uri uri  = getIntent().getData();
        String imgPath = ImageDegreeHelper.getPath(this,uri);
        int degree = ImageDegreeHelper.readPictureDegree(imgPath);
        Bitmap originBitmap = BitmapFactory.decodeFile(imgPath);
        if (originBitmap != null) {
            Bitmap bitmap = ImageDegreeHelper.rotaingImageView(degree, originBitmap.copy(Bitmap.Config.ARGB_8888, true));
            if (bitmap != null) {
                imgPhoto.setImageBitmap(bitmap);
                initMindspore(bitmap);
            }
        }
    }

    private void initMindspore(Bitmap bitmap) {
        try {
            trackingMobile = new ObjectTrackingMobile(this);
        } catch (FileNotFoundException e) {
            Log.e(TAG, Log.getStackTraceString(e));
            e.printStackTrace();
        }
        // 加载模型
        boolean ret = trackingMobile.loadModelFromBuf(getAssets());

        if (!ret) {
            Log.e(TAG, "Load model error.");
            return;
        }
        // run net.
        long startTime = System.currentTimeMillis();
        String result = trackingMobile.MindSpore_runnet(bitmap);
        long endTime = System.currentTimeMillis();

        Log.d(TAG, "RUNNET 耗时："+(endTime-startTime)+"ms");
        Log.d(TAG, "result："+ result);

        recognitionObjectBeanList = getRecognitionList(result);

        if (recognitionObjectBeanList != null && recognitionObjectBeanList.size() > 0) {
            drawRect(bitmap);
        }
    }

    private void drawRect(Bitmap bitmap) {
        Canvas canvas = new Canvas(bitmap);
        Paint mPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        mPaint.setTextSize(DisplayUtil.sp2px(this,30));
        //只绘制图形轮廓(描边)
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeWidth(DisplayUtil.dip2px(this,2));

        for (int i = 0; i < recognitionObjectBeanList.size(); i++) {
            RecognitionObjectBean objectBean = recognitionObjectBeanList.get(i);
            StringBuilder sb = new StringBuilder();
            sb.append(objectBean.getRectID()).append("_").append(objectBean.getObjectName()).append("_").append(String.format("%.2f", (100 * objectBean.getScore())) + "%");

            int paintColor =getResources().getColor(COLORS[i % COLORS.length]);
            mPaint.setColor(paintColor);

            RectF rectF = new RectF(objectBean.getLeft(), objectBean.getTop(), objectBean.getRight(), objectBean.getBottom());
            canvas.drawRect(rectF, mPaint);
            canvas.drawText(sb.toString(),objectBean.getLeft(), objectBean.getTop()-10,mPaint);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        trackingMobile.unloadModel();
    }

}