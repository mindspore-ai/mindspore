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

package com.mindspore.imageobject.imageclassification.ui;

import android.content.Context;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.Nullable;

import com.mindspore.imageobject.R;


public class HorTextView extends LinearLayout {
    private TextView tvLeftTitle;

    public TextView getTvRightContent() {
        return tvRightContent;
    }

    private TextView tvRightContent;
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

    public TextView getTvLeftTitle() {
        return tvLeftTitle;
    }
}
