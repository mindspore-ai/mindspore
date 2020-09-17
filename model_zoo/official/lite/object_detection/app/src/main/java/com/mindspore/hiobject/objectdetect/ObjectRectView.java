package com.mindspore.hiobject.objectdetect;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import com.mindspore.hiobject.help.RecognitionObjectBean;

import java.util.ArrayList;
import java.util.List;

/**
 * 针对物体检测的矩形框绘制类
 * <p>
 * 使用的API：
 * 1. Canvas：代表“依附”于指定View的画布，用它的方法来绘制各种图形.
 * 2. Paint：代表Canvas上的画笔，用于设置画笔颜色、画笔粗细、填充风格等.
 */

public class ObjectRectView extends View {

    private final String TAG = "ObjectRectView";

    private List<RecognitionObjectBean> mRecognitions = new ArrayList<>();
    private Paint mPaint = null;

    // 画框区域.
    private RectF mObjRectF;


    public ObjectRectView(Context context) {
        super(context);
        initialize();
    }

    public ObjectRectView(Context context, AttributeSet attrs) {
        super(context, attrs);
        initialize();
    }

    public ObjectRectView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        initialize();
    }


    public int[] MyColor = {Color.RED, Color.WHITE, Color.YELLOW, Color.GREEN, Color.LTGRAY, Color.MAGENTA, Color.BLACK, Color.BLUE, Color.CYAN};


    private void initialize() {
        mObjRectF = new RectF();

        mPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        mPaint.setTextSize(50);
        //只绘制图形轮廓(描边)
        mPaint.setStyle(Style.STROKE);
        mPaint.setStrokeWidth(5);
    }

    /**
     * 传入需绘制信息
     *
     * @param recognitions
     */
    public void setInfo(List<RecognitionObjectBean> recognitions) {
        Log.i(TAG, "setInfo: "+recognitions.size());

        mRecognitions.clear();
        mRecognitions.addAll(recognitions);

        //重新draw().
        invalidate();
    }

    public void clearCanvas(){
        mRecognitions.clear();
        //重新draw().
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (mRecognitions == null || mRecognitions.size() == 0) {
//            mPaint.setColor(Color.TRANSPARENT);
//            mObjRectF = new RectF(0, 0, 5, 5);
//            canvas.drawRoundRect(mObjRectF, 0, 0, mPaint);
            return;
        }
        for (int i = 0;i<mRecognitions.size();i++){
            RecognitionObjectBean bean = mRecognitions.get(i);
            mPaint.setColor(MyColor[i % MyColor.length]);
            drawRect(bean, canvas);
        }
    }



    public void drawRect(RecognitionObjectBean bean, Canvas canvas) {
        StringBuilder sb = new StringBuilder();
        sb.append(bean.getRectID()).append("_").append(bean.getObjectName()).append("_").append(String.format("%.2f", (100 * bean.getScore())) + "%");

        mObjRectF = new RectF(bean.getLeft(), bean.getTop(), bean.getRight(), bean.getBottom());
        canvas.drawRoundRect(mObjRectF, 5, 5, mPaint);
        canvas.drawText(sb.toString(), mObjRectF.left, mObjRectF.top -20 , mPaint);
    }
}
