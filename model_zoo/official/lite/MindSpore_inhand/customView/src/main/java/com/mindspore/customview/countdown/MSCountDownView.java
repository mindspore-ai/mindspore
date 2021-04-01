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
package com.mindspore.customview.countdown;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.AnimatorSet;
import android.animation.ValueAnimator;
import android.annotation.TargetApi;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Build;
import android.text.TextUtils;
import android.util.AttributeSet;
import android.view.View;
import android.view.animation.LinearInterpolator;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.mindspore.common.utils.DisplayUtil;
import com.mindspore.customview.R;

public class MSCountDownView extends View {

    private static final int HALF = 2;
    private Context mContext;
    private Paint mPaintBackGround, mPaintArc, mPaintText;

    private int mRetreatType;
    private float mPaintArcWidth;
    private int mCircleRadius;
    private int mPaintArcColor = Color.RED;
    private int mPaintBackGroundColor = Color.BLUE;
    private int mLoadingTime;
    private String mLoadingTimeUnit = "";
    private int mTextColor = Color.BLACK;
    private int mTextSize;
    private int location;
    private float startAngle, mmSweepAngleStart, mmSweepAngleEnd, mSweepAngle;

    private String mText = "";
    private int mWidth, mHeight;
    private AnimatorSet set;

    public MSCountDownView(Context context) {
        this(context, null);
    }

    public MSCountDownView(Context context, @Nullable AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public MSCountDownView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);

        mContext = context;
        init(attrs);
        initView();
    }

    private void init(AttributeSet attrs) {
        TypedArray array = mContext.obtainStyledAttributes(attrs, R.styleable.MSCountDownView);
        mRetreatType = array.getInt(R.styleable.MSCountDownView_ms_cd_retreat_type, 1);
        location = array.getInt(R.styleable.MSCountDownView_ms_cd_location, 1);
        mCircleRadius = (int) array.getDimension(R.styleable.MSCountDownView_ms_cd_circle_radius, DisplayUtil.dp2px(mContext, 25));
        mPaintArcWidth = array.getDimension(R.styleable.MSCountDownView_ms_cd_arc_width, DisplayUtil.dp2px(mContext, 3));
        mPaintArcColor = array.getColor(R.styleable.MSCountDownView_ms_cd_arc_color, mPaintArcColor);
        mTextSize = (int) array.getDimension(R.styleable.MSCountDownView_ms_cd_text_size, DisplayUtil.dp2px(mContext, 14));
        mTextColor = array.getColor(R.styleable.MSCountDownView_ms_cd_text_color, mTextColor);
        mPaintBackGroundColor = array.getColor(R.styleable.MSCountDownView_ms_cd_bg_color, mPaintBackGroundColor);
        mLoadingTime = array.getInteger(R.styleable.MSCountDownView_ms_cd_animator_time, 3);
        mLoadingTimeUnit = array.getString(R.styleable.MSCountDownView_ms_cd_animator_time_unit);
        if (TextUtils.isEmpty(mLoadingTimeUnit)) {
            mLoadingTimeUnit = "";
        }
        array.recycle();
    }

    @TargetApi(Build.VERSION_CODES.JELLY_BEAN)
    private void initView() {
        this.setBackground(ContextCompat.getDrawable(mContext, android.R.color.transparent));
        mPaintBackGround = new Paint();
        mPaintBackGround.setAntiAlias(true);
        mPaintBackGround.setStyle(Paint.Style.FILL);
        mPaintBackGround.setColor(mPaintBackGroundColor);

        mPaintArc = new Paint();
        mPaintArc.setAntiAlias(true);
        mPaintArc.setStyle(Paint.Style.STROKE);
        mPaintArc.setColor(mPaintArcColor);
        mPaintArc.setStrokeWidth(mPaintArcWidth);

        mPaintText = new Paint(Paint.ANTI_ALIAS_FLAG);
        mPaintText.setAntiAlias(true);
        mPaintText.setStyle(Paint.Style.STROKE);
        mPaintText.setStyle(Paint.Style.FILL);
        mPaintText.setColor(mTextColor);
        mPaintText.setTextSize(mTextSize);

        if (mLoadingTime < 0) {
            mLoadingTime = 3;
        }
        switch (location) {
            case 1:
                startAngle = -180;
                break;
            case 2:
                startAngle = -90;
                break;
            case 3:
                startAngle = 0;
                break;
            case 4:
                startAngle = 90;
                break;
        }

        if (mRetreatType == 1) {
            mmSweepAngleStart = 0f;
            mmSweepAngleEnd = 360f;
        } else {
            mmSweepAngleStart = 360f;
            mmSweepAngleEnd = 0f;
        }
    }


    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        mWidth = w;
        mHeight = h;
    }


    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
        setMeasuredDimension(mCircleRadius * HALF, mCircleRadius * HALF);
    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawCircle(mWidth / HALF, mHeight / HALF, mWidth / HALF - mPaintArcWidth, mPaintBackGround);
        RectF rectF = new RectF( mPaintArcWidth / HALF,  mPaintArcWidth / HALF, mWidth - mPaintArcWidth / HALF, mHeight - mPaintArcWidth / HALF);
        canvas.drawArc(rectF, startAngle, mSweepAngle, false, mPaintArc);
        float mTextWidth = mPaintText.measureText(mText, 0, mText.length());
        float dx = mWidth / HALF - mTextWidth / HALF;
        Paint.FontMetricsInt fontMetricsInt = mPaintText.getFontMetricsInt();
        float dy = (fontMetricsInt.bottom - fontMetricsInt.top) / HALF - fontMetricsInt.bottom;
        float baseLine = mHeight / HALF + dy;
        canvas.drawText(mText, dx, baseLine, mPaintText);
    }

    public void start() {
        ValueAnimator viewAnimator = ValueAnimator.ofFloat(mmSweepAngleStart, mmSweepAngleEnd);
        viewAnimator.setInterpolator(new LinearInterpolator());
        viewAnimator.addUpdateListener(valueAnimator -> {
            mSweepAngle = (float) valueAnimator.getAnimatedValue();
            invalidate();
        });
        ValueAnimator textAnimator = ValueAnimator.ofInt(mLoadingTime, 0);
        textAnimator.setInterpolator(new LinearInterpolator());
        textAnimator.addUpdateListener(valueAnimator -> {
            int time = (int) valueAnimator.getAnimatedValue();
            mText = time + mLoadingTimeUnit;
        });
        set = new AnimatorSet();
        set.setDuration(mLoadingTime * 1000);
        set.playTogether(viewAnimator, textAnimator);
        set.setInterpolator(new LinearInterpolator());
        set.start();
        set.addListener(new AnimatorListenerAdapter() {
            @Override
            public void onAnimationEnd(Animator animation) {
                super.onAnimationEnd(animation);
                clearAnimation();
                if (onCountDownFinishListener != null) {
                    onCountDownFinishListener.finish();
                }
            }
        });
    }

    public void stop() {
        onCountDownFinishListener = null;
        if (set != null && set.isRunning()) {
            set.cancel();
        }
    }

    public void setTime(int time) {
        mLoadingTime = time;
    }

    private OnCountDownFinishListener onCountDownFinishListener;

    public void setOnLoadingFinishListener(OnCountDownFinishListener listener) {
        this.onCountDownFinishListener = listener;
    }

    public interface OnCountDownFinishListener {
        void finish();
    }


}
