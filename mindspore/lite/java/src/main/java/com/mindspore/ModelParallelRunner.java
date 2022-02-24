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

package com.mindspore;

import com.mindspore.config.RunnerConfig;

import java.util.ArrayList;
import java.util.List;

/**
 * ModelParallelRunner is used to define a MindSpore ModelPoolManager, facilitating Model management.
 *
 * @since v1.6
 */
public class ModelParallelRunner {
    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private long modelParallelRunnerPtr;

    /**
     * Construct function.
     */
    public ModelParallelRunner() {
        this.modelParallelRunnerPtr = 0L;
    }

    /**
     * Get modelParallelRunnerPtr pointer.
     *
     * @return modelParallelRunnerPtr pointer.
     */
    public long getModelParallelRunnerPtr() {
        return this.modelParallelRunnerPtr;
    }

    /**
     * Build a model runner from model path so that it can run on a device. Only valid for Lite.
     *
     * @param modelPath    the model path.
     * @param runnerConfig the RunnerConfig Object.
     * @return init status.
     */
    public boolean init(String modelPath, RunnerConfig runnerConfig) {
        if (runnerConfig == null || modelPath == null) {
            return false;
        }
        modelParallelRunnerPtr = this.init(modelPath, runnerConfig.getRunnerConfigPtr());
        return modelParallelRunnerPtr != 0L;
    }

    /**
     * Build a model runner from model path so that it can run on a device. Only valid for Lite.
     *
     * @param modelPath the model path.
     * @return init status.
     */
    public boolean init(String modelPath) {
        if (modelPath == null) {
            return false;
        }
        modelParallelRunnerPtr = this.init(modelPath, 0L);
        return modelParallelRunnerPtr != 0;
    }

    /**
     * Build a model runner from model path so that it can run on a device. Only valid for Lite.
     *
     * @param inputs  inputs A vector where model inputs are arranged in sequence.
     * @param outputs outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
     * @return init status.
     */
    public boolean predict(List<MSTensor> inputs, List<MSTensor> outputs) {
        long[] inputsPtrArray = new long[inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            inputsPtrArray[i] = inputs.get(i).getMSTensorPtr();
        }
        List<Long> outputPtrs = predict(modelParallelRunnerPtr, inputsPtrArray);
        if (outputPtrs.isEmpty()) {
            return false;
        }
        for (int i = 0; i < outputPtrs.size(); i++) {
            if (outputPtrs.get(i) == 0L) {
                return false;
            }
            MSTensor msTensor = new MSTensor(outputPtrs.get(i));
            outputs.add(msTensor);
        }
        return true;
    }

    /**
     * Obtains all input tensors of the model.
     *
     * @return The vector that includes all input tensors.
     */
    public List<MSTensor> getInputs() {
        List<Long> ret = this.getInputs(this.modelParallelRunnerPtr);
        List<MSTensor> tensors = new ArrayList<>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Obtains all output tensors of the model.
     *
     * @return The vector that includes all input tensors.
     */
    public List<MSTensor> getOutputs() {
        List<Long> ret = this.getOutputs(this.modelParallelRunnerPtr);
        List<MSTensor> tensors = new ArrayList<>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Free model
     */
    public void free() {
        if (modelParallelRunnerPtr != 0L) {
            this.free(modelParallelRunnerPtr);
            modelParallelRunnerPtr = 0L;
        }
    }

    private native long init(String modelPath, long runnerConfigPtr);

    private native List<Long> predict(long modelParallelRunnerPtr, long[] inputs);

    private native List<Long> getInputs(long modelParallelRunnerPtr);

    private native List<Long> getOutputs(long modelParallelRunnerPtr);

    private native void free(long modelParallelRunnerPtr);
}
