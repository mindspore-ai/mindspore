/*
 * Copyright 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.mindspore.hms.camera;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.huawei.hms.mlsdk.common.MLAnalyzer;
import com.huawei.hms.mlsdk.text.MLText;
import com.huawei.hms.mlsdk.translate.cloud.MLRemoteTranslator;

public class MLSenceTextDetectionGraphic extends GraphicOverlay.Graphic {

    private final Context mContext;

    private final GraphicOverlay overlay;

    private final MLAnalyzer.Result<MLText.Block> results;

    private MLRemoteTranslator mlRemoteTranslator;

    public MLSenceTextDetectionGraphic(GraphicOverlay overlay, MLAnalyzer.Result<MLText.Block> results, Context context) {
        super(overlay);
        this.overlay = overlay;
        this.results = results;
        this.mContext = context;

    }

    @Override
    public void draw(Canvas canvas) {
        float x = overlay.getWidth() / 4.3f;
        float y = overlay.getHeight() - 550;
        Paint paint = new Paint();
        paint.setColor(Color.WHITE);
        paint.setTextSize(48);

        if (results.getAnalyseList().size() > 1) {
            for (int i = 0; i < results.getAnalyseList().size(); i++) {
                canvas.drawText(null == results.getAnalyseList().get(i) || null == results.getAnalyseList().get(i).getStringValue() ?
                                "" : results.getAnalyseList().get(i).getStringValue(),
                        x, y + 100 * (i + 1), paint);

            }
        }

        /*MLRemoteTranslateSetting setting = new MLRemoteTranslateSetting
                .Factory()
                .setTargetLangCode(StringUtils.isChinese(result) ? "en" : "zh")
                .create();


        mlRemoteTranslator = MLTranslatorFactory.getInstance().getRemoteTranslator(setting);

        Task<String> task = mlRemoteTranslator.asyncTranslate(result);
        task.addOnSuccessListener(new OnSuccessListener<String>() {
            @Override
            public void onSuccess(String text) {
                mTextView.setText(text);
            }

        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(Exception e) {

            }
        });*/
    }



}
