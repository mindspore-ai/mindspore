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

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;

import com.huawei.hms.mlsdk.gesture.MLGesture;

import java.util.List;

/**
 * Graphic instance for rendering hand position, orientation, and landmarks within an associated
 * graphic overlay view.
 *
 * @since 2020-12-16
 */
public class HandGestureGraphic extends GraphicOverlay.Graphic {

    private final List<MLGesture> results;

    private final Paint circlePaint;
    private final Paint textPaint;
    private final Paint linePaint;
    private final Paint rectPaint;
    private final Rect rect;

    public HandGestureGraphic(GraphicOverlay overlay, List<MLGesture> results) {
        super(overlay);

        this.results = results;

        circlePaint = new Paint();
        circlePaint.setColor(Color.RED);
        circlePaint.setStyle(Paint.Style.FILL);
        circlePaint.setAntiAlias(true);

        textPaint = new Paint();
        textPaint.setColor(Color.YELLOW);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setStrokeWidth(5f);
        textPaint.setTextSize(100);

        linePaint = new Paint();
        linePaint.setColor(Color.GREEN);
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(4f);
        linePaint.setAntiAlias(true);

        rectPaint = new Paint();
        rectPaint.setColor(Color.BLUE);
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(5f);
        rectPaint.setAntiAlias(true);

        rect = new Rect();

    }

    @Override
    public void draw(Canvas canvas) {
        for (int i = 0; i < results.size(); i++) {
            MLGesture mlGesture = results.get(i);

            canvas.drawRect(rect, rectPaint);

            Rect rect = translateRect(mlGesture.getRect());
            if (rect.right < rect.left) {
                int x = rect.left;
                rect.left = rect.right;
                rect.right = x;
            }
            canvas.drawRect(rect, linePaint);
            canvas.drawText(getChineseDescription(mlGesture.getCategory()),
                    translateX((mlGesture.getRect().left + mlGesture.getRect().right) / 2f),
                    translateY((mlGesture.getRect().top + mlGesture.getRect().bottom) / 2f),
                    textPaint);

        }

    }

    private String getChineseDescription(int gestureCategory) {
        String chineseDescription;
        switch (gestureCategory) {
            case MLGesture.ONE:
                chineseDescription = "数字1";
                break;
            case MLGesture.SECOND:
                chineseDescription = "数字2";
                break;
            case MLGesture.THREE:
                chineseDescription = "数字3";
                break;
            case MLGesture.FOUR:
                chineseDescription = "数字4";
                break;
            case MLGesture.FIVE:
                chineseDescription = "数字5";
                break;
            case MLGesture.SIX:
                chineseDescription = "数字6";
                break;
            case MLGesture.SEVEN:
                chineseDescription = "数字7";
                break;
            case MLGesture.EIGHT:
                chineseDescription = "数字8";
                break;
            case MLGesture.NINE:
                chineseDescription = "数字9";
                break;
            case MLGesture.DISS:
                chineseDescription = "差评";
                break;
            case MLGesture.FIST:
                chineseDescription = "握拳";
                break;
            case MLGesture.GOOD:
                chineseDescription = "点赞";
                break;
            case MLGesture.HEART:
                chineseDescription = "单手比心";
                break;
            case MLGesture.OK:
                chineseDescription = "确认";
                break;
            default:
                chineseDescription = "其他手势";
                break;

        }
        return chineseDescription;
    }

    public Rect translateRect(Rect rect) {
        float left = translateX(rect.left);
        float right = translateX(rect.right);
        float bottom = translateY(rect.bottom);
        float top = translateY(rect.top);
        if (left > right) {
            float size = left;
            left = right;
            right = size;
        }
        if (bottom < top) {
            float size = bottom;
            bottom = top;
            top = size;
        }
        return new Rect((int) left, (int) top, (int) right, (int) bottom);
    }
}
