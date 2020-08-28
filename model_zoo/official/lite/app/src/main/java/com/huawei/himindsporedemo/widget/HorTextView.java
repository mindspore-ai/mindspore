package com.huawei.himindsporedemo.widget;

import android.content.Context;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.Nullable;

import com.huawei.himindsporedemo.R;

public class HorTextView extends LinearLayout {
    private TextView tvLeftTitle, tvRightContent;
    private View viewBottomLine;

    public HorTextView(Context context) {
        this(context, null);
    }

    public HorTextView(Context context, @Nullable AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public HorTextView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        LayoutInflater.from(context).inflate(R.layout.layout_hor_text_view, this);
        tvLeftTitle = findViewById(R.id.tv_left_title);
        tvRightContent = findViewById(R.id.tv_right_content);
        viewBottomLine = findViewById(R.id.view_bottom_line);
    }


    public void setLeftTitle(String title) {
        tvLeftTitle.setText(title);
    }

    public void setRightContent(String content) {
        tvRightContent.setText(content);
    }

    public void setBottomLineVisible(int isVisible) {
        viewBottomLine.setVisibility(isVisible);
    }

}
