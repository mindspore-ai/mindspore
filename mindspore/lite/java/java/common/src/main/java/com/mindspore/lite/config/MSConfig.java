/*
 * Copyright 2020 Huawei Technologies Co., Ltd
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

package com.mindspore.lite.config;

/**
 * MSConfig is defined for holding environment variables during runtime.
 *
 * @since v1.0
 */
public class MSConfig {
    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private long msConfigPtr;

    public MSConfig() {
        this.msConfigPtr = 0;
    }

    /**
     * Initialize MSConfig.
     *
     * @param deviceType      A DeviceType enum type.
     * @param threadNum       Thread number config for thread pool.
     * @param cpuBindMode     A CpuBindMode enum variable.
     * @param isEnableFloat16 Whether to use float16 operator for priority.
     * @return Whether the initialization is successful.
     */
    public boolean init(int deviceType, int threadNum, int cpuBindMode, boolean isEnableFloat16) {
        this.msConfigPtr = createMSConfig(deviceType, threadNum, cpuBindMode, isEnableFloat16);
        return this.msConfigPtr != 0;
    }

    /**
     * Initialize MSConfig.
     *
     * @param deviceType  A DeviceType enum type.
     * @param threadNum   Thread number config for thread pool.
     * @param cpuBindMode A CpuBindMode enum variable.
     * @return Whether the initialization is successful.
     */
    public boolean init(int deviceType, int threadNum, int cpuBindMode) {
        this.msConfigPtr = createMSConfig(deviceType, threadNum, cpuBindMode, false);
        return this.msConfigPtr != 0;
    }

    /**
     * Initialize MSConfig.
     *
     * @param deviceType A DeviceType enum type.
     * @param threadNum  Thread number config for thread pool.
     * @return Whether the initialization is successful.
     */
    public boolean init(int deviceType, int threadNum) {
        return init(deviceType, threadNum, CpuBindMode.MID_CPU);
    }

    /**
     * Initialize MSConfig.
     *
     * @param deviceType A DeviceType enum type.
     * @return Whether the initialization is successful.
     */
    public boolean init(int deviceType) {
        return init(deviceType, 2);
    }

    /**
     * Initialize MSConfig.
     *
     * @return Whether the initialization is successful.
     */
    public boolean init() {
        return init(DeviceType.DT_CPU);
    }

    /**
     * Free all temporary memory in MindSpore Lite MSConfig.
     */
    public void free() {
        this.free(this.msConfigPtr);
        this.msConfigPtr = 0;
    }

    /**
     * return msconfig pointer
     *
     * @return msconfig pointer
     */
    public long getMSConfigPtr() {
        return msConfigPtr;
    }

    private native long createMSConfig(int deviceType, int threadNum, int cpuBindMode, boolean isEnableFloat16);

    private native void free(long msConfigPtr);
}
