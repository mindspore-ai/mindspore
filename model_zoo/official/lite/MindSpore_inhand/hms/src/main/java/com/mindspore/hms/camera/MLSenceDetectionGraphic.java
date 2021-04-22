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
import android.util.SparseArray;

import com.huawei.hms.mlsdk.scd.MLSceneDetection;
import com.mindspore.hms.R;

import java.text.NumberFormat;

import static com.mindspore.hms.scenedetection.SenceDetectionStillAnalyseActivity.mHashMap;

public class MLSenceDetectionGraphic extends GraphicOverlay.Graphic {
    private final GraphicOverlay overlay;

    private final SparseArray<MLSceneDetection> results;

    private final Context mContext;

    public MLSenceDetectionGraphic(GraphicOverlay overlay, SparseArray<MLSceneDetection> results, Context context) {
        super(overlay);
        this.overlay = overlay;
        this.results = results;
        this.mContext = context;
    }

    @Override
    public void draw(Canvas canvas) {
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setTextSize(48);


        canvas.drawText(mContext.getResources().getString(R.string.image_scene_count) + "：" + results.size(), overlay.getWidth() / 5f, 50, paint);
        for (int i = 0; i < results.size(); i++) {
            String result = results.get(i).getResult().toLowerCase();
            if (mHashMap != null) {
                if (mHashMap.containsKey(result)) {
                    result = mHashMap.get(result);
                }
            }
            canvas.drawText(mContext.getResources().getString(R.string.image_scene) + "：" + result, overlay.getWidth() / 5f, 100 * (i + 1), paint);
            NumberFormat fmt = NumberFormat.getPercentInstance();
            fmt.setMaximumFractionDigits(2);
            float confidence = results.get(i).getConfidence();
            String format = fmt.format(confidence);
            canvas.drawText( mContext.getResources().getString(R.string.image_score) + "：" + format, overlay.getWidth() / 5f, (100 * (i + 1)) + 50, paint);

        }
    }
}
