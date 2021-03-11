package com.mindspore.himindspore.ui.view;

import android.content.Context;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.widget.RelativeLayout;
import androidx.annotation.Nullable;

import com.mindspore.himindspore.R;

public class LeftImageRightTextButton extends RelativeLayout {

    public LeftImageRightTextButton(Context context)
    {
        super(context, null);
    }

    public LeftImageRightTextButton(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        LayoutInflater.from(context).inflate(R.layout.btn_left_iamge_right_text, this,true);
    }
}
