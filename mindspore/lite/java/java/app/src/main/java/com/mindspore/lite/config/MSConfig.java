/**
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

public class MSConfig {
    private long msConfigPtr;

    public MSConfig() {
        this.msConfigPtr = 0;
    }

    public boolean init(int deviceType, int threadNum, int cpuBindMode, boolean enable_float16) {
        this.msConfigPtr = createMSConfig(deviceType, threadNum, cpuBindMode, enable_float16);
        return this.msConfigPtr != 0;
    }

    public boolean init(int deviceType, int threadNum, int cpuBindMode) {
        this.msConfigPtr = createMSConfig(deviceType, threadNum, cpuBindMode, false);
        return this.msConfigPtr != 0;
    }

    public boolean init(int deviceType, int threadNum) {
        return init(deviceType, threadNum, CpuBindMode.MID_CPU);
    }

    public boolean init(int deviceType) {
        return init(deviceType, 2);
    }

    public boolean init() {
        return init(DeviceType.DT_CPU);
    }

    public void free() {
        this.free(this.msConfigPtr);
        this.msConfigPtr = 0;
    }

    public long getMSConfigPtr() {
        return msConfigPtr;
    }

    private native long createMSConfig(int deviceType, int threadNum, int cpuBindMode, boolean enable_float16);

    private native void free(long msConfigPtr);
}
