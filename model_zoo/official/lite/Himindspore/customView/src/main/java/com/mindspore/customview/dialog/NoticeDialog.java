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
package com.mindspore.customview.dialog;

import android.app.Dialog;
import android.content.Context;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;

import com.mindspore.common.utils.DisplayUtil;
import com.mindspore.customview.R;

public class NoticeDialog extends Dialog {

    private Context context;
    private TextView title;
    private TextView content;

    private TextView bottomText;
    private ImageView bottomImage;
    private View bottomLine;

    private String titleString, contentString, bottomString;
    private boolean isShowBottomLogo = false;
    private int gravity = Gravity.CENTER;

    public NoticeDialog setGravity(int gravity) {
        this.gravity = gravity;
        return this;
    }

    public void setYesOnclickListener(YesOnclickListener yesOnclickListener) {
        this.yesOnclickListener = yesOnclickListener;
    }

    private YesOnclickListener yesOnclickListener;


    public NoticeDialog setTitleString(String titleString) {
        this.titleString = titleString;
        return this;
    }


    public NoticeDialog setContentString(String contentString) {
        this.contentString = contentString;
        return this;
    }

    public NoticeDialog setShowBottomLogo(boolean showBottomLogo) {
        this.isShowBottomLogo = showBottomLogo;
        return this;
    }

    public NoticeDialog(@NonNull Context context) {
        super(context, R.style.NoticeDialog);
        this.context = context;

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.layout_dialog_notice);
        setCanceledOnTouchOutside(false);
        initView();
        initData();
        initEvent();
    }


    private void initView() {
        title = findViewById(R.id.dialog_title);
        content = findViewById(R.id.dialog_content);
        bottomText = findViewById(R.id.dialog_confirm);
        bottomImage = findViewById(R.id.img_dialog_bottom);
        bottomLine = findViewById(R.id.line_bottom);
    }

    private void initData() {
        if (!TextUtils.isEmpty(titleString)) {
            title.setText(titleString);
        }
        if (!TextUtils.isEmpty(contentString)) {
            content.setText(contentString);
        }
        if (!TextUtils.isEmpty(bottomString)) {
            bottomText.setText(bottomString);
        }
        if (isShowBottomLogo) {
            bottomImage.setVisibility(View.VISIBLE);
            LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, DisplayUtil.dp2px(context,1));
            layoutParams.setMargins(DisplayUtil.dp2px(context,22),DisplayUtil.dp2px(context,22), DisplayUtil.dp2px(context,22), 0);
            bottomLine.setLayoutParams(layoutParams);
        } else {
            bottomImage.setVisibility(View.GONE);
        }

        content.setGravity(gravity);
    }


    private void initEvent() {
        bottomText.setOnClickListener(view -> {
            if (yesOnclickListener != null) {
                yesOnclickListener.onYesOnclick();
            }
        });
    }


    public interface YesOnclickListener {
        void onYesOnclick();
    }

}
