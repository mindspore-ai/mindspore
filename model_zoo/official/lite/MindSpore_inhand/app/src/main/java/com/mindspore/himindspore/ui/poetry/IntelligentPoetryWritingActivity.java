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
package com.mindspore.himindspore.ui.poetry;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.widget.ProgressBar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.himindspore.R;
@Route(path = "/app/IntelligentPoetryWritingActivity")
public class IntelligentPoetryWritingActivity extends AppCompatActivity {

    private WebView mWebView;
    private ProgressBar progressBar;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_intelligent_poetry_writing);
        progressBar = findViewById(R.id.progress);
        initView();
    }


    @SuppressLint("JavascriptInterface")
    private void initView() {
        findViewById(R.id.w_back).setOnClickListener(v -> finish());
        mWebView = findViewById(R.id.mWebView);
        WebSettings wSet = mWebView.getSettings();
        wSet.setJavaScriptEnabled(true);
        wSet.setDomStorageEnabled(true);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            mWebView.getSettings().setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        }
        mWebView.setWebChromeClient(new WebChromeClient(){
            @Override
            public void onProgressChanged(WebView view, int newProgress) {
                if(newProgress==100){
                    progressBar.setVisibility(View.GONE);
                }
                else{
                    progressBar.setVisibility(View.VISIBLE);
                    progressBar.setProgress(newProgress);
                }

            }
        });
//        mWebView.loadUrl("http://114.116.235.161/resources/tech/nlp/poetry");
        mWebView.loadUrl(MSLinkUtils.HELP_INTELLIGENT_POETRY);
        mWebView.addJavascriptInterface(IntelligentPoetryWritingActivity.this,"android");
    }

    @JavascriptInterface
    public void getGeneratePoetry(final String text){
        Intent intent = new Intent(IntelligentPoetryWritingActivity.this,PoetryPosterActivity.class);
        intent.putExtra("POEM",text);
        startActivity(intent);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mWebView.removeAllViews();
        mWebView.destroy();
    }
}