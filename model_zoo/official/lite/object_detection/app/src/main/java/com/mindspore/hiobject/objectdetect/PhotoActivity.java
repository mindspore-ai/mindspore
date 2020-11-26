package com.mindspore.hiobject.objectdetect;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.hiobject.R;
import com.mindspore.hiobject.help.ImageDegreeHelper;
import com.mindspore.hiobject.help.RecognitionObjectBean;
import com.mindspore.hiobject.help.TrackingMobile;

import java.io.FileNotFoundException;
import java.util.List;

import static com.mindspore.hiobject.help.RecognitionObjectBean.getRecognitionList;

public class PhotoActivity extends AppCompatActivity {

    private static final String TAG = "PhotoActivity";
    private static final int[] COLORS ={Color.RED, Color.WHITE, Color.YELLOW, Color.GREEN, Color.LTGRAY, Color.MAGENTA, Color.BLACK, Color.BLUE, Color.CYAN};

    private ImageView imgPhoto;
    private TrackingMobile trackingMobile;
    private List<RecognitionObjectBean> recognitionObjectBeanList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_photo);

        imgPhoto = findViewById(R.id.img_photo);

        Uri uri  = getIntent().getData();
        String imgPath = ImageDegreeHelper.getPath(this,uri);
        int degree = ImageDegreeHelper.readPictureDegree(imgPath);
        Bitmap originBitmap = BitmapFactory.decodeFile(imgPath);
        if (originBitmap != null) {
            Bitmap bitmap = ImageDegreeHelper.rotaingImageView(degree, originBitmap.copy(Bitmap.Config.ARGB_8888, true));
            if (bitmap != null) {
                Matrix matrix = new Matrix();
                matrix.setScale(0.7f, 0.7f);
                bitmap = Bitmap.createBitmap( bitmap, 0, 0,  bitmap.getWidth(), bitmap.getHeight(), matrix, false);

                imgPhoto.setImageBitmap(bitmap);
                initMindspore(bitmap);
            }
        }
    }

    private void initMindspore(Bitmap bitmap) {
        try {
            trackingMobile = new TrackingMobile(this);
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

        Log.d(TAG, "RUNNET CONSUMING："+(endTime-startTime)+"ms");
        Log.d(TAG, "result："+ result);

        recognitionObjectBeanList = getRecognitionList(result);

        if (recognitionObjectBeanList != null && recognitionObjectBeanList.size() > 0) {
            drawRect(bitmap);
        }
    }

    private void drawRect(Bitmap bitmap) {
        Canvas canvas = new Canvas(bitmap);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setTextSize(dip2px(15));
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(3);

        for (int i = 0; i < recognitionObjectBeanList.size(); i++) {
            RecognitionObjectBean objectBean = recognitionObjectBeanList.get(i);
            StringBuilder sb = new StringBuilder();
            sb.append(objectBean.getRectID()).append("_").append(objectBean.getObjectName()).append("_").append(String.format("%.2f", (100 * objectBean.getScore())) + "%");

            int paintColor = COLORS[i % COLORS.length];
            paint.setColor(paintColor);

            RectF rectF = new RectF(objectBean.getLeft(), objectBean.getTop(), objectBean.getRight(), objectBean.getBottom());
            canvas.drawRect(rectF, paint);
            canvas.drawText(sb.toString(),objectBean.getLeft(), objectBean.getTop()-10,paint);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        trackingMobile.unloadModel();
    }

    public  int dip2px(float dipValue){
        float scale = getResources().getDisplayMetrics().density;
        return (int) (dipValue*scale+0.5f);

    }
}