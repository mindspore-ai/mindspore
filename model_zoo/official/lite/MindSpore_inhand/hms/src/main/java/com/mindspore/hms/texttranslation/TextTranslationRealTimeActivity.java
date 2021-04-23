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
package com.mindspore.hms.texttranslation;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.mindspore.hms.R;
import com.mindspore.hms.camera.GraphicOverlay;
import com.mindspore.hms.camera.LensEnginePreview;

@Route(path = "/hms/TextTranslationRealTimeActivity")
public class TextTranslationRealTimeActivity extends AppCompatActivity {

    private LensEnginePreview mPreview;

    private GraphicOverlay mOverlay;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text_translation_real_time);
        init();
    }
    private void init(){
        Toolbar mToolbar = findViewById(R.id.TextVideo_activity_toolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
        this.mPreview = this.findViewById(R.id.scene_preview);
        this.mOverlay = this.findViewById(R.id.scene_overlay);
    }
}