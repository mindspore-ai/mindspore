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

import android.graphics.Canvas;

import com.huawei.hms.mlsdk.common.LensEngine;


public abstract class BaseGraphic {
    private final GraphicOverlay graphicOverlay;

    public BaseGraphic(GraphicOverlay overlay) {
        this.graphicOverlay = overlay;
    }

    public abstract void draw(Canvas canvas);

    public float scaleX(float x) {
        return x * this.graphicOverlay.getWidthScaleValue();
    }

    public float scaleY(float y) {
        return y * this.graphicOverlay.getHeightScaleValue();
    }

    public float translateX(float x) {
        if (this.graphicOverlay.getCameraFacing() == LensEngine.FRONT_LENS) {
            return this.graphicOverlay.getWidth() - this.scaleX(x);
        } else {
            return this.scaleX(x);
        }
    }

    public float translateY(float y) {
        return this.scaleY(y);
    }
}
