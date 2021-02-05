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
package com.mindspore.styletransfer;

import android.graphics.Bitmap;

public class ModelExecutionResult {
    private Bitmap styledImage;
    private long preProcessTime;
    private long stylePredictTime;
    private long styleTransferTime;
    private long postProcessTime;
    private long totalExecutionTime;
    private String executionLog;
    private String errorMessage;


    public ModelExecutionResult(Bitmap styledImage, long preProcessTime, long stylePredictTime, long styleTransferTime, long postProcessTime, long totalExecutionTime, String executionLog) {
        this.styledImage = styledImage;
        this.preProcessTime = preProcessTime;
        this.stylePredictTime = stylePredictTime;
        this.styleTransferTime = styleTransferTime;
        this.postProcessTime = postProcessTime;
        this.totalExecutionTime = totalExecutionTime;
        this.executionLog = executionLog;
    }

    public Bitmap getStyledImage() {
        return styledImage;
    }

    public ModelExecutionResult setStyledImage(Bitmap styledImage) {
        this.styledImage = styledImage;
        return this;
    }

    public long getPreProcessTime() {
        return preProcessTime;
    }

    public ModelExecutionResult setPreProcessTime(long preProcessTime) {
        this.preProcessTime = preProcessTime;
        return this;
    }

    public long getStylePredictTime() {
        return stylePredictTime;
    }

    public ModelExecutionResult setStylePredictTime(long stylePredictTime) {
        this.stylePredictTime = stylePredictTime;
        return this;
    }

    public long getStyleTransferTime() {
        return styleTransferTime;
    }

    public ModelExecutionResult setStyleTransferTime(long styleTransferTime) {
        this.styleTransferTime = styleTransferTime;
        return this;
    }

    public long getPostProcessTime() {
        return postProcessTime;
    }

    public ModelExecutionResult setPostProcessTime(long postProcessTime) {
        this.postProcessTime = postProcessTime;
        return this;
    }

    public long getTotalExecutionTime() {
        return totalExecutionTime;
    }

    public ModelExecutionResult setTotalExecutionTime(long totalExecutionTime) {
        this.totalExecutionTime = totalExecutionTime;
        return this;
    }

    public String getExecutionLog() {
        return executionLog;
    }

    public ModelExecutionResult setExecutionLog(String executionLog) {
        this.executionLog = executionLog;
        return this;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public ModelExecutionResult setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
        return this;
    }
}
