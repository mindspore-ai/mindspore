package com.mindspore.himindspore.objectdetection.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;


import com.mindspore.himindspore.R;
import com.mindspore.himindspore.objectdetection.bean.RecognitionObjectBean;
import com.mindspore.himindspore.utils.DisplayUtil;

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

    private Context context;

    public ObjectRectView(Context context) {
        this(context,null);
    }

    public ObjectRectView(Context context, AttributeSet attrs) {
        this(context, attrs,0);
    }

    public ObjectRectView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        this.context =context;
        initialize();
    }


    private static final int[] MyColor ={R.color.white,R.color.text_blue,R.color.text_yellow,R.color.text_orange,R.color.text_green};


    private void initialize() {
        mObjRectF = new RectF();

        mPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        mPaint.setTextSize(DisplayUtil.sp2px(context,16));
        //只绘制图形轮廓(描边)
        mPaint.setStyle(Style.STROKE);
        mPaint.setStrokeWidth(DisplayUtil.dip2px(context,2));
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
            return;
        }
        for (int i = 0;i<mRecognitions.size();i++){
            RecognitionObjectBean bean = mRecognitions.get(i);
            mPaint.setColor(context.getResources().getColor(MyColor[i % MyColor.length]));
            drawRect(bean, canvas);
        }
    }



    public void drawRect(RecognitionObjectBean bean, Canvas canvas) {
        StringBuilder sb = new StringBuilder();
        sb.append(bean.getRectID()).append("_").append(bean.getObjectName()).append("_").append(String.format("%.2f", (100 * bean.getScore())) + "%");

        mObjRectF = new RectF(bean.getLeft(), bean.getTop(), bean.getRight(), bean.getBottom());
        canvas.drawRect(mObjRectF, mPaint);
        canvas.drawText(sb.toString(), mObjRectF.left, mObjRectF.top - DisplayUtil.dip2px(context,10) , mPaint);
    }

}
