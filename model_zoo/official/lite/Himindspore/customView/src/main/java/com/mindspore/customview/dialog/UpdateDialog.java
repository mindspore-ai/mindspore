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
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;

import com.mindspore.customview.R;

public class UpdateDialog extends Dialog {

    private Context context;
    private TextView title, content;


    private Button btnConfirm, btnCancle;
    private View bottomLine;

    private String titleString, contentString;


    private YesOnclickListener yesOnclickListener;
    private NoOnclickListener noOnclickListener;

    public Button getBtnConfirm() {
        return btnConfirm;
    }

    public Button getBtnCancle() {
        return btnCancle;
    }

    public void setYesOnclickListener(YesOnclickListener yesOnclickListener) {
        this.yesOnclickListener = yesOnclickListener;
    }

    public void setNoOnclickListener(NoOnclickListener noOnclickListener) {
        this.noOnclickListener = noOnclickListener;
    }


    public UpdateDialog setTitleString(String titleString) {
        this.titleString = titleString;
        return this;
    }


    public UpdateDialog setContentString(String contentString) {
        this.contentString = contentString;
        return this;
    }


    public UpdateDialog(@NonNull Context context) {
        super(context, R.style.NoticeDialog);
        this.context = context;

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.layout_dialog_update);
        setCanceledOnTouchOutside(false);
        initView();
        initData();
        initEvent();
    }


    private void initView() {
        title = findViewById(R.id.dialog_title);
        content = findViewById(R.id.dialog_content);
        bottomLine = findViewById(R.id.line_bottom);
        btnConfirm = findViewById(R.id.dialog_confirm);
        btnCancle = findViewById(R.id.dialog_cancle);
    }

    private void initData() {
        if (!TextUtils.isEmpty(titleString)) {
            title.setText(titleString);
        }
        if (!TextUtils.isEmpty(contentString)) {
            content.setText(contentString);
        }
    }

    private void initEvent() {
        btnConfirm.setOnClickListener(view -> {
            if (yesOnclickListener != null) {
                yesOnclickListener.onYesOnclick();
            }
        });
        btnCancle.setOnClickListener(view -> {
            if (noOnclickListener != null) {
                noOnclickListener.onNoOnclick();
            }
        });
    }


    public interface YesOnclickListener {
        void onYesOnclick();
    }

    public interface NoOnclickListener {
        void onNoOnclick();
    }

}
