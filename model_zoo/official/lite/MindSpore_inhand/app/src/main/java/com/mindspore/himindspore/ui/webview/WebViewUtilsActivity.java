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
package com.mindspore.himindspore.ui.webview;

import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ProgressBar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.himindspore.R;

public class WebViewUtilsActivity extends AppCompatActivity {

    private WebView mWebView;
    private ProgressBar progressBar;
    private Toolbar mToolbar;
    private String mWebViewUrl;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_web_view_utils);
        mWebViewUrl = getIntent().getStringExtra("WebView");
        initView();
    }

    private void initView() {
        progressBar = findViewById(R.id.progress);
        mToolbar = findViewById(R.id.mWebView_toolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
        mWebView = findViewById(R.id.mWebView);
        mWebView.setWebViewClient(new WebViewClient());
        mWebView.getSettings().setJavaScriptEnabled(true);
        mWebView.getSettings().setDomStorageEnabled(true);
        mWebView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onProgressChanged(WebView view, int newProgress) {

                if (newProgress == 100) {
                    progressBar.setVisibility(View.GONE);
                } else {
                    progressBar.setVisibility(View.VISIBLE);
                    progressBar.setProgress(newProgress);
                }
            }
        });
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            mWebView.getSettings().setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        }
        showWebViewTitle(mWebViewUrl);
        mWebView.loadUrl(mWebViewUrl);
    }

    private void showWebViewTitle(String mWebViewUrl) {
        switch (mWebViewUrl) {
            case MSLinkUtils.ME_STAR_URL:
                mToolbar.setTitle(R.string.me_up_title);
                break;
            case MSLinkUtils.BASE_URL:
                mToolbar.setTitle(R.string.me_official_title);
                break;
            case MSLinkUtils.ME_CODE_URL:
                mToolbar.setTitle(R.string.me_official_code_title);
                break;
            case MSLinkUtils.ME_HELP_URL:
                mToolbar.setTitle(R.string.me_qa_title);
                break;
            case MSLinkUtils.COLLEGE_QUICK_APP:
                mToolbar.setTitle(R.string.title_college_broken_side);
                break;
            case MSLinkUtils.COLLEGE_MAIN_FAQ:
                mToolbar.setTitle(R.string.title_college_faq);
                break;
            case MSLinkUtils.COLLEGE_MAIN_ASK:
                mToolbar.setTitle(R.string.title_college_forum);
                break;
            case MSLinkUtils.COLLEGE_MAIN_CLOUD:
                mToolbar.setTitle(R.string.title_college_one_hour);
                break;
            case MSLinkUtils.COLLEGE_QUICK_EXECUTE:
                mToolbar.setTitle(R.string.title_college_perform);
                break;
            case MSLinkUtils.COLLEGE_QUICK_VIDEO:
                mToolbar.setTitle(R.string.title_college_video);
                break;
            case MSLinkUtils.COLLEGE_QUICK_TRAIN:
                mToolbar.setTitle(R.string.title_college_training);
                break;
            case MSLinkUtils.USER_PRIVACY_RULES:
                mToolbar.setTitle(R.string.me_user_agreements);
                break;
            default:
                mToolbar.setTitle(R.string.me_official_title);
                break;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mWebView.removeAllViews();
        mWebView.destroy();
    }
}