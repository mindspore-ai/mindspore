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

package com.mindspore.hms.camera;

import android.hardware.Camera;

public class CameraConfiguration {

    public static final int CAMERA_FACING_BACK = Camera.CameraInfo.CAMERA_FACING_BACK;
    public static final int CAMERA_FACING_FRONT = Camera.CameraInfo.CAMERA_FACING_FRONT;

    public static final int DEFAULT_WIDTH = 480;
    public static final int DEFAULT_HEIGHT = 360;

    public static final int MAX_WIDTH = 960;
    public static final int MAX_HEIGHT = 720;
    protected static int cameraFacing = CameraConfiguration.CAMERA_FACING_BACK;

    private float fps = 20.0f;
    private int previewWidth = CameraConfiguration.MAX_WIDTH;
    private int previewHeight = CameraConfiguration.MAX_HEIGHT;
    private final boolean isAutoFocus = true;

    public synchronized void setCameraFacing(int facing) {
        if ((facing != CameraConfiguration.CAMERA_FACING_BACK) && (facing != CameraConfiguration.CAMERA_FACING_FRONT)) {
            throw new IllegalArgumentException("Invalid camera: " + facing);
        }
        cameraFacing = facing;
    }

    public float getFps() {
        return this.fps;
    }

    public void setFps(float fps) {
        this.fps = fps;
    }

    public int getPreviewWidth() {
        return this.previewWidth;
    }

    public void setPreviewWidth(int previewWidth) {
        this.previewWidth = previewWidth;
    }

    public int getPreviewHeight() {
        return this.previewHeight;
    }

    public void setPreviewHeight(int previewHeight) {
        this.previewHeight = previewHeight;
    }

    public boolean isAutoFocus() {
        return this.isAutoFocus;
    }

    public synchronized static int getCameraFacing() {
        return cameraFacing;
    }
}
