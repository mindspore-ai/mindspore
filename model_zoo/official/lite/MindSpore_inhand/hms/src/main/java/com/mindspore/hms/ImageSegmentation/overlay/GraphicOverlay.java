/**
 * Copyright 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.mindspore.hms.ImageSegmentation.overlay;

import android.content.Context;
import android.graphics.Canvas;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;


import com.mindspore.hms.camera.CameraConfiguration;

import java.util.ArrayList;
import java.util.List;

public class GraphicOverlay extends View {
    private final Object lock = new Object();
    private int previewWidth;
    private int previewHeight;
    private float widthScaleValue = 1.0f;
    private float heightScaleValue = 1.0f;
    private int cameraFacing = CameraConfiguration.CAMERA_FACING_BACK;
    private final List<BaseGraphic> graphics = new ArrayList<>();

    public GraphicOverlay(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public void clear() {
        synchronized (this.lock) {
            this.graphics.clear();
        }
        this.postInvalidate();
    }

    public void addGraphic(BaseGraphic graphic) {
        synchronized (this.lock) {
            this.graphics.add(graphic);
        }
    }

    public void removeGraphic(BaseGraphic graphic) {
        synchronized (this.lock) {
            this.graphics.remove(graphic);
        }
        this.postInvalidate();
    }

    public void setCameraInfo(int width, int height, int facing) {
        synchronized (this.lock) {
            this.previewWidth = width;
            this.previewHeight = height;
            this.cameraFacing = facing;
        }
        this.postInvalidate();
    }

    public int getCameraFacing() {
        return this.cameraFacing;
    }

    public float getWidthScaleValue() {
        return this.widthScaleValue;
    }

    public float getHeightScaleValue() {
        return this.heightScaleValue;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        synchronized (this.lock) {
            if ((this.previewWidth != 0) && (this.previewHeight != 0)) {
                this.widthScaleValue = (float) canvas.getWidth() / (float) this.previewWidth;
                this.heightScaleValue = (float) canvas.getHeight() / (float) this.previewHeight;
            }
            for (BaseGraphic graphic : this.graphics) {
                graphic.draw(canvas);
            }
        }
    }
}
