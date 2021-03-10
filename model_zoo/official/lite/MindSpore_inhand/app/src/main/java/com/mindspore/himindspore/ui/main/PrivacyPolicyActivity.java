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
package com.mindspore.himindspore.ui.main;

import android.os.Bundle;
import android.view.View;
import android.webkit.WebSettings;
import android.webkit.WebView;

import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.himindspore.R;

public class PrivacyPolicyActivity extends AppCompatActivity {

    private static final String TAG = PrivacyPolicyActivity.class.getSimpleName();

    private WebView mWebView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_privacy_policy);
        initView();
    }

    private void initView() {
        findViewById(R.id.w_back).setOnClickListener(v -> finish());
        mWebView = findViewById(R.id.mWebView);
        WebSettings wSet = mWebView.getSettings();
        wSet.setJavaScriptEnabled(true);
        mWebView.loadUrl(MSLinkUtils.USER_PRIVACY_RULES);

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mWebView.removeAllViews();
        mWebView.destroy();
    }
}