/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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

import com.mindspore.lite.NativeLibrary;

public class MSContext {
    static {
        try {
            NativeLibrary.load();
        } catch (Exception e) {
            System.err.println("Failed to load MindSporLite native library.");
            e.printStackTrace();
            throw e;
        }
    }

    private long msContextPtr;

    /**
     * Construct function.
     */
    public MSContext() {
        this.msContextPtr = 0;
    }

    /**
     * Add device info to context.
     *
     * @param deviceType support cpu,npu and gpu.
     * @param isEnableFloat16  whether to use float16 operator for priority.
     * @param npuFreq npu frequency used for npu device.
     * @return add status.
     */
    public boolean addDeviceInfo(int deviceType, boolean isEnableFloat16, int npuFreq) {
        return addDeviceInfo(msContextPtr, deviceType, isEnableFloat16, npuFreq);
    }

    /**
     * Add device info to context.
     *
     * @param deviceType support cpu,npu and gpu.
     * @param isEnableFloat16  whether to use float16 operator for priority.
     * @return add status.
     */
    public boolean addDeviceInfo(int deviceType, boolean isEnableFloat16) {
        return addDeviceInfo(msContextPtr, deviceType, isEnableFloat16, 3);
    }

    /**
     * Init Context,default use 2 thread,no bind mode.
     *
     * @return init status.
     */
    public boolean init() {
        this.msContextPtr = createMSContext(2, 0, false);
        return this.msContextPtr != 0;
    }

    /**
     * Init Context.
     *
     * @param threadNum thread nums.
     * @param cpuBindMode support bind high,mid cpu.0,no bind.1,bind mid cpu.2. bind high cpu.
     * @return init status.
     */
    public boolean init(int threadNum, int cpuBindMode) {
        this.msContextPtr = createMSContext(threadNum, cpuBindMode, false);
        return this.msContextPtr != 0;
    }

    /**
     * Init Context.
     *
     * @param threadNum thread nums.
     * @param cpuBindMode support bind high,mid cpu.0,no bind.1,bind mid cpu.2. bind high cpu.
     * @param isEnableParallel enable parallel in multi devices.
     * @return init status.
     */
    public boolean init(int threadNum, int cpuBindMode, boolean isEnableParallel) {
        this.msContextPtr = createMSContext(threadNum, cpuBindMode, isEnableParallel);
        return this.msContextPtr != 0;
    }

    /**
     * Free context.
     */
    public void free() {
        this.free(this.msContextPtr);
        this.msContextPtr = 0;
    }

    /**
     * Get context pointer.
     *
     * @return context pointer.
     */
    public long getMSContextPtr() {
        return msContextPtr;
    }

    private native long createMSContext(int threadNum, int cpuBindMode, boolean enableParallel);

    private native boolean addDeviceInfo(long msContextPtr, int deviceType, boolean isEnableFloat16, int npuFrequency);

    private native void free(long msContextPtr);
}