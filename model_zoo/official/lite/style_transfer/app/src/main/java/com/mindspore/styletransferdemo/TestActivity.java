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
package com.mindspore.styletransferdemo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;

public class TestActivity extends AppCompatActivity {

    private ImageView img1, img2, img3;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test);

        img1 = findViewById(R.id.img1);
        img2 = findViewById(R.id.img2);
        img3 = findViewById(R.id.img3);

        Bitmap bitmap1 = BitmapFactory.decodeResource(getResources(), R.drawable.person);
        Glide.with(this).load(bitmap1).into(img1);

        Bitmap bitmap2 =
                ImageUtils.loadBitmapFromResources(this, "thumbnails/style3.jpg");
        Glide.with(this).load(bitmap2).into(img2);


        StyleTransferModelExecutor transferModelExecutor = new StyleTransferModelExecutor(this, false);
        ModelExecutionResult result = transferModelExecutor.execute(bitmap1, bitmap2);
        Glide.with(this).load(result.getStyledImage()).into(img3);
    }
}