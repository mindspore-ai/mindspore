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

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.common.utils.ImageUtils;
import com.mindspore.himindspore.R;

public class PoetryPosterActivity extends AppCompatActivity {

    private Button saveBtn, shareBtn;
    private PoetryView poetryView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_poetry_poster);
        findViewById(R.id.w_back).setOnClickListener(v -> finish());
        String poem = getIntent().getStringExtra("POEM");
        saveBtn = findViewById(R.id.save_btn);
        shareBtn = findViewById(R.id.share_btn);
        poetryView = findViewById(R.id.top_layout);

        poetryView.setPoemText(poem);

        saveBtn.setOnClickListener(view -> {
            Uri imgPath = ImageUtils.saveToAlbum(PoetryPosterActivity.this, poetryView,null,false);
            if (imgPath != null) {
                Toast.makeText(PoetryPosterActivity.this, R.string.poem_save_success, Toast.LENGTH_SHORT).show();
            }
        });

        shareBtn.setOnClickListener(view -> {
            Uri imgPath = ImageUtils.saveToAlbum(PoetryPosterActivity.this, poetryView,"poemshare",false);
            Intent share_intent = new Intent();
            share_intent.setAction(Intent.ACTION_SEND);
            share_intent.setType("image/*");
            share_intent.putExtra(Intent.EXTRA_STREAM, imgPath);
            share_intent = Intent.createChooser(share_intent, getString(R.string.title_share));
            startActivity(share_intent);
        });
    }


}