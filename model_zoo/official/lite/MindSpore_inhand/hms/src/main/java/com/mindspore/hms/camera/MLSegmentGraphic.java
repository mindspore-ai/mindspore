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

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;

import com.huawei.hms.mlsdk.imgseg.MLImageSegmentation;

public class MLSegmentGraphic extends GraphicOverlay.Graphic {
    private static final String TAG = MLSegmentGraphic.class.getSimpleName();
    private final Rect mDestRect;
    private final Paint resultPaint;
    private final Bitmap bitmapForeground;
    private final Boolean isFront;

    public MLSegmentGraphic(LensEnginePreview preview, GraphicOverlay overlay, MLImageSegmentation segmentation, Boolean isFront) {
        super(overlay);
        this.bitmapForeground = segmentation.getForeground();
        this.isFront = isFront;

        int width = bitmapForeground.getWidth();
        int height = bitmapForeground.getHeight();
        int div = overlay.getWidth() - preview.getWidth();
        int left = overlay.getWidth() - width + div / 2;
        mDestRect = new Rect(left, 0, overlay.getWidth() - div / 2, height / 2);
        this.resultPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        this.resultPaint.setFilterBitmap(true);
        this.resultPaint.setDither(true);
    }

    @Override
    public void draw(Canvas canvas) {
        canvas.drawBitmap(isFront ? convert(bitmapForeground) : bitmapForeground, null, mDestRect, resultPaint);
    }


    private Bitmap convert(Bitmap bitmap) {
        Matrix m = new Matrix();
        m.setScale(-1, 1);
        Bitmap reverseBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), m, true);
        return reverseBitmap;
    }
}
