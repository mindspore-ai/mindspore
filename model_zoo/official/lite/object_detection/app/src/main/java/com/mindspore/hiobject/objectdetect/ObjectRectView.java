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
 * Rectangle drawing class for object detection
 *
 * 1. Canvas：Represents the canvas attached to the specified view and uses its method to draw various graphics
 * 2. Paint：Represents the brush on canvas and is used to set brush color, brush thickness, fill style, etc
 */

public class ObjectRectView extends View {

    private final String TAG = "ObjectRectView";

    private List<RecognitionObjectBean> mRecognitions = new ArrayList<>();
    private Paint mPaint = null;

    // Frame area
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
        //Draw only outline (stroke)
        mPaint.setStyle(Style.STROKE);
        mPaint.setStrokeWidth(5);
    }

    /**
     * Input information to be drawn
     *
     * @param recognitions
     */
    public void setInfo(List<RecognitionObjectBean> recognitions) {
        Log.i(TAG, "setInfo: "+recognitions.size());

        mRecognitions.clear();
        mRecognitions.addAll(recognitions);

        invalidate();
    }

    public void clearCanvas(){
        mRecognitions.clear();
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
