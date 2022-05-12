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

/**
 * Configuration for ModelParallelRunner.
 *
 * @since v1.6
 */
public class RunnerConfig {
    static {
        MindsporeLite.init();
    }

    private long runnerConfigPtr;

    /**
     * Construct function.
     */
    public RunnerConfig() {
        this.runnerConfigPtr = 0L;
    }

    /**
     * Init RunnerConfig
     *
     * @return init status.
     */
    public boolean init() {
        this.runnerConfigPtr = createRunnerConfig();
        return this.runnerConfigPtr != 0L;
    }

    /**
     * Init RunnerConfig
     *
     * @param msContext MSContext Object.
     * @return init status.
     */
    public boolean init(MSContext msContext) {
        if (msContext == null) {
            return false;
        }
        this.runnerConfigPtr = createRunnerConfigWithContext(msContext.getMSContextPtr());
        return this.runnerConfigPtr != 0L;
    }

    /**
     * Set workers num
     *
     * @param workersNum The number of parallel models.
     */
    public void setWorkersNum(int workersNum) {
        setWorkersNum(runnerConfigPtr, workersNum);
    }

    /**
     * Get RunnerConfig pointer.
     *
     * @return RunnerConfig pointer.
     */
    public long getRunnerConfigPtr() {
        return runnerConfigPtr;
    }

    /**
     * Fre RunnerConfig pointer.
     */
    public void free() {
        this.free(runnerConfigPtr);
        runnerConfigPtr = 0;
    }

    private native long createRunnerConfig();

    private native long createRunnerConfigWithContext(long msContextPtr);

    private native void setWorkersNum(long runnerConfigPtr, int workersNum);

    private native boolean free(long runnerConfigPtr);
}
