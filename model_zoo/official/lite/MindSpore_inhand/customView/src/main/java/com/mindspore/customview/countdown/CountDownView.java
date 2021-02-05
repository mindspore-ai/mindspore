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
import android.graphics.Typeface;
import android.os.Build;
import android.text.TextUtils;
import android.util.AttributeSet;
import android.view.View;
import android.view.animation.LinearInterpolator;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.mindspore.customview.R;

public class CountDownView extends View {

    private Context mContext;//上下文
    private Paint mPaintBackGround;//背景画笔
    private Paint mPaintArc;//圆弧画笔
    private Paint mPaintText;//文字画笔
    private int mRetreatType;//圆弧绘制方式（增加和减少）
    private float mPaintArcWidth;//最外层圆弧的宽度
    private int mCircleRadius;//圆圈的半径
    private int mPaintArcColor = Color.parseColor("#3C3F41");//初始值
    private int mPaintBackGroundColor = Color.parseColor("#55B2E5");//初始值
    private int mLoadingTime;//时间，单位秒
    private String mLoadingTimeUnit = "";//时间单位
    private int mTextColor = Color.BLACK;//字体颜色
    private int mTextSize;//字体大小
    private int location;//从哪个位置开始
    private float startAngle;//开始角度
    private float mmSweepAngleStart;//起点
    private float mmSweepAngleEnd;//终点
    private float mSweepAngle;//扫过的角度
    private String mText = "";//要绘制的文字
    private int mWidth;
    private int mHeight;
    private AnimatorSet set;

    public CountDownView(Context context) {
        this(context, null);
    }

    public CountDownView(Context context, @Nullable AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public CountDownView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        mContext = context;

        TypedArray array = context.obtainStyledAttributes(attrs, R.styleable.CountDownView);
        mRetreatType = array.getInt(R.styleable.CountDownView_cd_retreat_type, 1);
        location = array.getInt(R.styleable.CountDownView_cd_location, 1);
        mCircleRadius = (int) array.getDimension(R.styleable.CountDownView_cd_circle_radius, dip2px(context, 25));//默认25dp
        mPaintArcWidth = array.getDimension(R.styleable.CountDownView_cd_arc_width, dip2px(context, 3));//默认3dp
        mPaintArcColor = array.getColor(R.styleable.CountDownView_cd_arc_color, mPaintArcColor);
        mTextSize = (int) array.getDimension(R.styleable.CountDownView_cd_text_size, dip2px(context, 14));//默认14sp
        mTextColor = array.getColor(R.styleable.CountDownView_cd_text_color, mTextColor);
        mPaintBackGroundColor = array.getColor(R.styleable.CountDownView_cd_bg_color, mPaintBackGroundColor);
        mLoadingTime = array.getInteger(R.styleable.CountDownView_cd_animator_time, 3);//默认3秒
        mLoadingTimeUnit = array.getString(R.styleable.CountDownView_cd_animator_time_unit);//时间单位
        if (TextUtils.isEmpty(mLoadingTimeUnit)) {
            mLoadingTimeUnit = "";
        }
        array.recycle();
        init();
    }

    @TargetApi(Build.VERSION_CODES.JELLY_BEAN)
    private void init() {
        //背景设为透明，然后造成圆形View的视觉错觉
        this.setBackground(ContextCompat.getDrawable(mContext, android.R.color.transparent));
        mPaintBackGround = new Paint();
        mPaintBackGround.setStyle(Paint.Style.FILL);
        mPaintBackGround.setAntiAlias(true);
        mPaintBackGround.setColor(mPaintBackGroundColor);

        mPaintArc = new Paint();
        mPaintArc.setStyle(Paint.Style.STROKE);
        mPaintArc.setAntiAlias(true);
        mPaintArc.setColor(mPaintArcColor);
        mPaintArc.setStrokeWidth(mPaintArcWidth);

        mPaintText = new Paint(Paint.ANTI_ALIAS_FLAG);
        mPaintText.setStyle(Paint.Style.STROKE);
        mPaintText.setStyle(Paint.Style.FILL);
        mPaintText.setAntiAlias(true);
        mPaintText.setColor(mTextColor);
        mPaintText.setTextSize(mTextSize);
        //如果时间为小于0，则默认倒计时时间为3秒
        if (mLoadingTime < 0) {
            mLoadingTime = 3;
        }
        if (location == 1) {//默认从左侧开始
            startAngle = -180;
        } else if (location == 2) {
            startAngle = -90;
        } else if (location == 3) {
            startAngle = 0;
        } else if (location == 4) {
            startAngle = 90;
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
        //获取view宽高
        mWidth = w;
        mHeight = h;
    }


    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
        //因为必须是圆形的view，所以在这里重新赋值
        setMeasuredDimension(mCircleRadius * 2, mCircleRadius * 2);
    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        //画北景园
        canvas.drawCircle(mWidth / 2, mHeight / 2, mWidth / 2 - mPaintArcWidth, mPaintBackGround);
        //画圆弧
        RectF rectF = new RectF(0 + mPaintArcWidth / 2, 0 + mPaintArcWidth / 2, mWidth - mPaintArcWidth / 2, mHeight - mPaintArcWidth / 2);
        canvas.drawArc(rectF, startAngle, mSweepAngle, false, mPaintArc);
        //画文字
        float mTextWidth = mPaintText.measureText(mText, 0, mText.length());
        float dx = mWidth / 2 - mTextWidth / 2;
        Paint.FontMetricsInt fontMetricsInt = mPaintText.getFontMetricsInt();
        float dy = (fontMetricsInt.bottom - fontMetricsInt.top) / 2 - fontMetricsInt.bottom;
        float baseLine = mHeight / 2 + dy;
        canvas.drawText(mText, dx, baseLine, mPaintText);
    }

    /**
     * 开始动态倒计时
     */
    public void start() {
        ValueAnimator animator = ValueAnimator.ofFloat(mmSweepAngleStart, mmSweepAngleEnd);
        animator.setInterpolator(new LinearInterpolator());
        animator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() {
            @Override
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                mSweepAngle = (float) valueAnimator.getAnimatedValue();
                //获取到需要绘制的角度，重新绘制
                invalidate();
            }
        });
        //这里是时间获取和赋值
        ValueAnimator animator1 = ValueAnimator.ofInt(mLoadingTime, 0);
        animator1.setInterpolator(new LinearInterpolator());
        animator1.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() {
            @Override
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                int time = (int) valueAnimator.getAnimatedValue();
                mText = time + mLoadingTimeUnit;
            }
        });
        set = new AnimatorSet();
        set.playTogether(animator, animator1);
        set.setDuration(mLoadingTime * 1000);
        set.setInterpolator(new LinearInterpolator());
        set.start();
        set.addListener(new AnimatorListenerAdapter() {
            @Override
            public void onAnimationEnd(Animator animation) {
                super.onAnimationEnd(animation);
                clearAnimation();
                if (loadingFinishListener != null) {
                    loadingFinishListener.finish();
                }
            }
        });
    }

    /**
     * 停止动画
     */
    public void stop() {
        loadingFinishListener = null;

        if (set != null && set.isRunning()) {
            set.cancel();
        }
    }

    /**
     * 设置倒计时时间
     *
     * @param time 时间，秒
     */
    public void setTime(int time) {
        mLoadingTime = time;
    }

    private OnLoadingFinishListener loadingFinishListener;

    public void setOnLoadingFinishListener(OnLoadingFinishListener listener) {
        this.loadingFinishListener = listener;
    }

    public interface OnLoadingFinishListener {
        void finish();
    }

    /**
     * 根据手机的分辨率从 dp 的单位 转成为 px(像素)
     */
    public static int dip2px(Context context, float dpValue) {
        final float scale = context.getResources().getDisplayMetrics().density;
        return (int) (dpValue * scale + 0.5f);
    }
}
