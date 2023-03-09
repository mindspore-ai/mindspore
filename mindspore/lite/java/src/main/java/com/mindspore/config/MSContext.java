/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Context is used to store environment variables during execution.
 *
 * @since v1.0
 */
public class MSContext {
    private static Logger LOGGER = Logger.getLogger(MSContext.class.toString());
    static {
        MindsporeLite.init();
    }

    private static final long EMPTY_CONTEXT_PTR_VALUE = 0L;
    private static final int ERROR_VALUE = -1;
    private static final String NULLPTR_ERROR_MESSAGE="Context pointer from java is nullptr.";

    private long msContextPtr;

    /**
     * Construct function.
     */
    public MSContext() {
        this.msContextPtr = EMPTY_CONTEXT_PTR_VALUE;
    }

    /**
     * Add device info to context.
     *
     * @param deviceType      support cpu,npu and gpu.
     * @param isEnableFloat16 whether to use float16 operator for priority.
     * @param npuFreq         npu frequency used for npu device.
     * @return add status.
     */
    public boolean addDeviceInfo(int deviceType, boolean isEnableFloat16, int npuFreq) {
        return addDeviceInfo(msContextPtr, deviceType, isEnableFloat16, npuFreq);
    }

    /**
     * Add device info to context.
     *
     * @param deviceType      support cpu,npu and gpu.
     * @param isEnableFloat16 whether to use float16 operator for priority.
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
        this.msContextPtr = createDefaultMSContext();
        return this.msContextPtr != EMPTY_CONTEXT_PTR_VALUE;
    }

    /**
     * Init Context.
     *
     * @param threadNum   thread nums.
     * @param cpuBindMode support bind high,mid cpu.0,no bind.1,bind mid cpu.2. bind high cpu.
     * @return init status.
     */
    public boolean init(int threadNum, int cpuBindMode) {
        this.msContextPtr = createMSContext(threadNum, cpuBindMode, false);
        return this.msContextPtr != EMPTY_CONTEXT_PTR_VALUE;
    }

    /**
     * Init Context.
     *
     * @param threadNum        thread nums.
     * @param cpuBindMode      support bind high,mid cpu.0,no bind.1,bind mid cpu.2. bind high cpu.
     * @param isEnableParallel enable parallel in multi devices.
     * @return init status.
     */
    public boolean init(int threadNum, int cpuBindMode, boolean isEnableParallel) {
        this.msContextPtr = createMSContext(threadNum, cpuBindMode, isEnableParallel);
        return this.msContextPtr != EMPTY_CONTEXT_PTR_VALUE;
    }

    /**
     * Free context.
     */
    public void free() {
        if (isInitialized()) {
            this.free(this.msContextPtr);
            this.msContextPtr = EMPTY_CONTEXT_PTR_VALUE;
        } else {
            LOGGER.log(Level.SEVERE, NULLPTR_ERROR_MESSAGE);
        }
    }

    /**
     * Get context pointer.
     *
     * @return context pointer.
     */
    public long getMSContextPtr() {
        return msContextPtr;
    }

    /**
     * Check weather the msContextPtr has been initialized or not.
     *
     * @return true: initialized; false: not initialized.
     */
    private boolean isInitialized() {
        return this.msContextPtr != EMPTY_CONTEXT_PTR_VALUE;
    }

    /**
     * set the number of threads at runtime. 
     * If you haven't init context before, it will do nothing.
     *
     * @param threadNum the number of threads at runtime.
     */
    public void setThreadNum(int threadNum) {
        if (isInitialized()) {
            setThreadNum(this.msContextPtr, threadNum);
        } else {
            LOGGER.log(Level.SEVERE, NULLPTR_ERROR_MESSAGE);
        }
    }

    /**
     * get the current thread number setting. 
     * If you haven't init context, it will return {@value  ERROR_VALUE}.
     *
     * @return The current thread number setting.
     */
    public int getThreadNum() {
        int retVal = ERROR_VALUE;
        if (isInitialized()) {
            retVal = getThreadNum(this.msContextPtr);
        } else {
            LOGGER.log(Level.SEVERE, NULLPTR_ERROR_MESSAGE);
        }
        return retVal;
    }

    /**
     * set the parallel number of operators at runtime. 
     * If you haven't init context before, it will do nothing.
     *
     * @param parallelNum parallelNum the parallel number of operators at runtime.
     */
    public void setInterOpParallelNum(int parallelNum) {
        if (isInitialized()) {
            setInterOpParallelNum(this.msContextPtr, parallelNum);
        } else {
            LOGGER.log(Level.SEVERE, NULLPTR_ERROR_MESSAGE);
        }
    }

    /**
     * get the current operators parallel number setting. 
     * If you haven't init context, it will return {@value  ERROR_VALUE}.
     *
     * @return The current operators parallel number setting.
     */
    public int getInterOpParallelNum() {
        int retVal = ERROR_VALUE;
        if (isInitialized()) {
            retVal = getInterOpParallelNum(this.msContextPtr);
        } else {
            LOGGER.log(Level.SEVERE, NULLPTR_ERROR_MESSAGE);
        }
        return retVal;
    }

    /**
     * set the thread affinity to CPU cores. 
     * If you haven't init context before, it will do nothing.
     *
     * @param mode: 0: no affinities, 1: big cores first, 2: little cores first
     */
    public void setThreadAffinity(int mode) {
        if (isInitialized()) {
            setThreadAffinity(this.msContextPtr, mode);
        } else {
            LOGGER.log(Level.SEVERE, NULLPTR_ERROR_MESSAGE);
        }
    }


    /**
     * get the thread affinity of CPU cores. 
     * If you haven't init context, it will return {@value  ERROR_VALUE}.
     *
     * @return Thread affinity to CPU cores. 0: no affinities, 1: big cores first, 2: little cores first
     */
    public int getThreadAffinityMode() {
        int retVal = ERROR_VALUE;
        if (isInitialized()) {
            retVal = getThreadAffinityMode(this.msContextPtr);
        } else {
            LOGGER.log(Level.SEVERE, NULLPTR_ERROR_MESSAGE);
        }
        return retVal;
    }

    /**
     * set the thread lists to CPU cores. 
     * If coreList and mode are set by setThreadAffinity at the same time, the coreList is effective, but the
     * mode is not effective.
     * If you haven't init context before, it will do nothing.
     *
     * @param coreList An {@code ArrayList<Integer>} of thread core lists.
     */
    public void setThreadAffinity(ArrayList<Integer> coreList) {
        if (isInitialized()) {
            int len = coreList.size();
            int[] coreListArray = new int[len];
            for (int i = 0; i < len; i++) {
                coreListArray[i] = coreList.get(i);
            }
            setThreadAffinity(this.msContextPtr, coreListArray);
        } else {
            LOGGER.log(Level.SEVERE, NULLPTR_ERROR_MESSAGE);
        }
    }

    /**
     * get the thread lists of CPU cores. 
     * If you haven't init context, it will return {@value  ERROR_VALUE}.
     *
     * @return An {@code ArrayList<Integer>} of thread core lists.
     */

    public ArrayList<Integer> getThreadAffinityCoreList() {
        ArrayList<Integer> retVal = new ArrayList<>();
        if (isInitialized()) {
            retVal = getThreadAffinityCoreList(this.msContextPtr);
        } else {
            LOGGER.log(Level.SEVERE, NULLPTR_ERROR_MESSAGE);
        }
        return retVal;
    }

    /**
     * set the status whether to perform model inference or training in parallel. 
     * If you haven't init context before, it will do nothing.
     *
     * @param isParallel: true, parallel; false, not in parallel.
     */
    public void setEnableParallel(boolean isParallel) {
        if (isInitialized()) {
            setEnableParallel(this.msContextPtr, isParallel);
        } else {
            LOGGER.log(Level.SEVERE, NULLPTR_ERROR_MESSAGE);
        }
    }

    /**
     * get the status whether to perform model inference or training in parallel. 
     * If you haven't init context, it will also return false.
     *
     * @return boolean value that indicates whether in parallel.
     */
    public boolean getEnableParallel() {
        boolean retVal = false;
        if (isInitialized()) {
            retVal = getEnableParallel(this.msContextPtr);
        } else {
            LOGGER.log(Level.SEVERE, NULLPTR_ERROR_MESSAGE);
        }
        return retVal;
    }


    private native long createMSContext(int threadNum, int cpuBindMode, boolean enableParallel);

    private native long createDefaultMSContext();

    private native boolean addDeviceInfo(long msContextPtr, int deviceType, boolean isEnableFloat16, int npuFrequency);

    private native void free(long msContextPtr);

    private native void setThreadNum(long msContextPtr, int threadNum);

    private native int getThreadNum(long msContextPtr);

    private native void setInterOpParallelNum(long msContextPtr, int parallelNum);

    private native int getInterOpParallelNum(long msContextPtr);

    private native void setThreadAffinity(long msContextPtr, int mode);

    private native int getThreadAffinityMode(long msContextPtr);

    private native void setThreadAffinity(long msContextPtr, int[] coreList);

    private native ArrayList<Integer> getThreadAffinityCoreList(long msContextPtr);

    private native void setEnableParallel(long msContextPtr, boolean isParallel);

    private native boolean getEnableParallel(long msContextPtr);
}