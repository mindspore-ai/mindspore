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

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import com.mindspore.imageobject.R;
import com.mindspore.imageobject.objectdetection.bean.RecognitionObjectBean;
import com.mindspore.imageobject.util.DisplayUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * Rectangle drawing class for object detection
 * <p>
 * 1. Canvas：Represents the canvas attached to the specified view and uses its method to draw various graphics
 * 2. Paint：Represents the brush on canvas and is used to set brush color, brush thickness, fill style, etc
 */

public class ObjectRectView extends View {

    private final String TAG = "ObjectRectView";

    private List<RecognitionObjectBean> mRecognitions = new ArrayList<>();
    private Paint mPaint = null;
    private Paint mPaintText = null;

    // Frame area
    private RectF mObjRectF;

    private Context context;

    public ObjectRectView(Context context) {
        this(context, null);
    }

    public ObjectRectView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public ObjectRectView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        this.context = context;
        initialize();
    }


    private static final int[] MyColor = {R.color.white, R.color.text_blue, R.color.text_yellow, R.color.text_orange, R.color.text_green};


    private void initialize() {
        mObjRectF = new RectF();

        mPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        //Draw only outline (stroke)
        mPaint.setStyle(Style.STROKE);
        mPaint.setStrokeWidth(DisplayUtil.dip2px(context, 2));

        mPaintText = new Paint(Paint.ANTI_ALIAS_FLAG);
        mPaintText.setTextSize(DisplayUtil.sp2px(context, 16));
        mPaintText.setStyle(Paint.Style.FILL);
    }

    /**
     * Input information to be drawn
     *
     * @param recognitions
     */
    public void setInfo(List<RecognitionObjectBean> recognitions) {
        Log.i(TAG, "setInfo: " + recognitions.size());

        mRecognitions.clear();
        mRecognitions.addAll(recognitions);
        postInvalidate();
    }

    public void clearCanvas() {
        mRecognitions.clear();
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (mRecognitions == null || mRecognitions.size() == 0) {
            return;
        }
        for (int i = 0; i < mRecognitions.size(); i++) {
            RecognitionObjectBean bean = mRecognitions.get(i);
            mPaint.setColor(context.getResources().getColor(MyColor[i % MyColor.length]));
            mPaintText.setColor(context.getResources().getColor(MyColor[i % MyColor.length]));
            drawRect(bean, canvas);
        }
    }


    public void drawRect(RecognitionObjectBean bean, Canvas canvas) {
        StringBuilder sb = new StringBuilder();
        sb.append(bean.getRectID()).append("—").append(bean.getObjectName()).append("—").append(String.format("%.2f", (100 * bean.getScore())) + "%");

        mObjRectF = new RectF(bean.getLeft(), bean.getTop(), bean.getRight(), bean.getBottom());
        canvas.drawRect(mObjRectF, mPaint);
        if (mObjRectF.top < DisplayUtil.dip2px(context, 20)){
            canvas.drawText(sb.toString(), mObjRectF.left + DisplayUtil.dip2px(context, 5), mObjRectF.top + DisplayUtil.dip2px(context, 20), mPaintText);
        }else{
            canvas.drawText(sb.toString(), mObjRectF.left, mObjRectF.top - DisplayUtil.dip2px(context, 10), mPaintText);
        }
    }

}
