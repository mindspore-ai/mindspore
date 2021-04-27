package com.mindspore.himindspore.ui.view;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;

import com.mindspore.himindspore.R;

public class IconButtonStyleView extends RelativeLayout {

    private RelativeLayout mBtnView;
    private ImageView mBtnImage;
    private TextView mBtnTextName, mBtnTextAngle;
    private CharSequence text_name, text_angle;
    private Drawable btn_icon;

    public IconButtonStyleView(Context context) {
        super(context);
    }

    public IconButtonStyleView(Context context, AttributeSet attrs) {
        super(context, attrs);
        TypedArray btnStyle = context.obtainStyledAttributes(attrs, R.styleable.iconButtonStyle);
        btn_icon = btnStyle.getDrawable(R.styleable.iconButtonStyle_buttonImage);
        text_name = btnStyle.getText(R.styleable.iconButtonStyle_buttonTextName);
        text_angle = btnStyle.getText(R.styleable.iconButtonStyle_buttonTextAngle);

        btnStyle.recycle();
    }

    @Override
    protected void onFinishInflate() {
        super.onFinishInflate();
        mBtnView = (RelativeLayout) LayoutInflater.from(getContext()).inflate(R.layout.button_style_layout, this, true);

        mBtnImage = mBtnView.findViewById(R.id.btnImage);
        mBtnTextName = mBtnView.findViewById(R.id.textName);
        mBtnTextAngle = mBtnView.findViewById(R.id.textAngle);

        mBtnImage.setImageDrawable(btn_icon);
        mBtnTextName.setText(text_name);
        mBtnTextAngle.setText(text_angle);
    }
}
