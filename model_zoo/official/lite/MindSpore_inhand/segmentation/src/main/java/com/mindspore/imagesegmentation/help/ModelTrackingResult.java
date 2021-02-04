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
package com.mindspore.imagesegmentation.help;

import android.graphics.Bitmap;

import java.util.Set;

public class ModelTrackingResult {
    private Bitmap bitmapResult;
    private Bitmap bitmapOriginal;
    private Bitmap bitmapMaskOnly;
    private String executionLog;
    private Set<Integer> itemsFound;

    public ModelTrackingResult(Bitmap bitmapResult, Bitmap bitmapOriginal, Bitmap bitmapMaskOnly, String executionLog, Set<Integer> itemsFound) {
        this.bitmapResult = bitmapResult;
        this.bitmapOriginal = bitmapOriginal;
        this.bitmapMaskOnly = bitmapMaskOnly;
        this.executionLog = executionLog;
        this.itemsFound = itemsFound;
    }

    public Bitmap getBitmapResult() {
        return bitmapResult;
    }

    public ModelTrackingResult setBitmapResult(Bitmap bitmapResult) {
        this.bitmapResult = bitmapResult;
        return this;
    }

    public Bitmap getBitmapOriginal() {
        return bitmapOriginal;
    }

    public ModelTrackingResult setBitmapOriginal(Bitmap bitmapOriginal) {
        this.bitmapOriginal = bitmapOriginal;
        return this;
    }

    public Bitmap getBitmapMaskOnly() {
        return bitmapMaskOnly;
    }

    public ModelTrackingResult setBitmapMaskOnly(Bitmap bitmapMaskOnly) {
        this.bitmapMaskOnly = bitmapMaskOnly;
        return this;
    }

    public String getExecutionLog() {
        return executionLog;
    }

    public ModelTrackingResult setExecutionLog(String executionLog) {
        this.executionLog = executionLog;
        return this;
    }

    public Set<Integer> getItemsFound() {
        return itemsFound;
    }

    public ModelTrackingResult setItemsFound(Set<Integer> itemsFound) {
        this.itemsFound = itemsFound;
        return this;
    }
}
