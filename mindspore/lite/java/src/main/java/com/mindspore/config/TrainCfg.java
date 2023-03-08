/*
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mindspore.config;

/**
 * TrainCfg Class
 *
 * @since v1.0
 */
public class TrainCfg {
    static {
        try {
            System.loadLibrary("mindspore-lite-train-jni");
        } catch (Exception e) {
            System.err.println("Failed to load MindSporLite native library.");
            e.printStackTrace();
            throw e;
        }
    }

    private long trainCfgPtr;

    /**
     * Construct function.
     */
    public TrainCfg() {
        this.trainCfgPtr = 0;
    }

    /**
     * Init train config.
     *
     * @return init status.
     */
    public boolean init() {
        this.trainCfgPtr = createTrainCfg(null, 0, false);
        return this.trainCfgPtr != 0;
    }

    /**
     * Init train config specified loss name.
     *
     * @param loss_name loss name used for split inference and train part.
     * @return init status.
     */
    public boolean init(String loss_name) {
        this.trainCfgPtr = createTrainCfg(loss_name, 0, false);
        return this.trainCfgPtr != 0;
    }

    /**
     * Free train config.
     */
    public void free() {
        this.free(this.trainCfgPtr);
        this.trainCfgPtr = 0;
    }

    /**
     * Add mix precision config to train config.
     *
     * @param dynamicLossScale if dynamic or fix loss scale factor.
     * @param lossScale        loss scale factor.
     * @param thresholdIterNum a threshold for modifying loss scale when dynamic loss scale is enabled.
     * @return add status.
     */
    public boolean addMixPrecisionCfg(boolean dynamicLossScale, float lossScale, int thresholdIterNum) {
        return addMixPrecisionCfg(trainCfgPtr, dynamicLossScale, lossScale, thresholdIterNum);
    }

    /**
     * Get train config pointer.
     *
     * @return train config pointer.
     */
    public long getTrainCfgPtr() {
        return trainCfgPtr;
    }

    private native long createTrainCfg(String lossName, int optimizationLevel, boolean accmulateGrads);

    private native boolean addMixPrecisionCfg(long trainCfgPtr, boolean dynamicLossScale, float lossScale,
                                              int thresholdIterNum);

    private native void free(long trainCfgPtr);
}