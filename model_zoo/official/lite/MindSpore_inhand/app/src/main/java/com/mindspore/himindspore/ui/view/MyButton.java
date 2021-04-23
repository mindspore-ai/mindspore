package com.mindspore.himindspore.ui.view;

import android.content.Context;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.widget.RelativeLayout;
import androidx.annotation.Nullable;

import com.mindspore.himindspore.R;

public class MyButton extends RelativeLayout {

    public MyButton(Context context)
    {
        super(context, null);
    }

    public MyButton(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        LayoutInflater.from(context).inflate(R.layout.my_button, this,true);
    }
}
