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

import com.mindspore.config.MindsporeLite;
import com.mindspore.config.RunnerConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * ModelParallelRunner is used to define a MindSpore ModelPoolManager, facilitating Model management.
 *
 * @since v1.6
 */
public class ModelParallelRunner {
    static {
        MindsporeLite.init();
    }

    private long modelParallelRunnerPtr;
    private AtomicLong signal = new AtomicLong();
    private ReentrantReadWriteLock rwLock = new ReentrantReadWriteLock();

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
     * Build a model runner from model path so that it can run on a device.
     *
     * @param modelPath    the model path.
     * @param runnerConfig the RunnerConfig Object.
     * @return init status.
     */
    public boolean init(String modelPath, RunnerConfig runnerConfig) {
        rwLock.writeLock().lock();
        if (runnerConfig == null || modelPath == null) {
            rwLock.writeLock().unlock();
            return false;
        }
        if (modelParallelRunnerPtr != 0L){
            rwLock.writeLock().unlock();
            return true;
        }
        modelParallelRunnerPtr = this.init(modelPath, runnerConfig.getRunnerConfigPtr());
        rwLock.writeLock().unlock();
        return modelParallelRunnerPtr != 0L;
    }

    /**
     * Build a model runner from model path so that it can run on a device.
     *
     * @param modelPath the model path.
     * @return init status.
     */
    public boolean init(String modelPath) {
        rwLock.writeLock().lock();
        if (modelPath == null) {
            rwLock.writeLock().unlock();
            return false;
        }
        if (modelParallelRunnerPtr != 0L){
            rwLock.writeLock().unlock();
            return true;
        }
        modelParallelRunnerPtr = this.init(modelPath, 0L);
        rwLock.writeLock().unlock();
        return modelParallelRunnerPtr != 0;
    }

    /**
     * Build a model runner from model path so that it can run on a device.
     *
     * @param inputs  inputs A vector where model inputs are arranged in sequence.
     * @param outputs outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
     * @return init status.
     */
    public boolean predict(List<MSTensor> inputs, List<MSTensor> outputs) {
        signal.incrementAndGet();
        rwLock.readLock().lock();
        try {
            if (this.modelParallelRunnerPtr == 0L) {
                return false;
            }
            if (inputs == null || outputs == null || inputs.size() == 0) {
                return false;
            }
            long[] inputsPtrArray = new long[inputs.size()];
            for (int i = 0; i < inputs.size(); i++) {
                inputsPtrArray[i] = inputs.get(i).getMSTensorPtr();
            }
            if(outputs.size() != 0){
                long[] outputsPtrArray = new long[outputs.size()];
                for (int i = 0; i < outputs.size(); i++) {
                    if(outputs.get(i) == null){
                        return false;
                    }
                    outputsPtrArray[i] = outputs.get(i).getMSTensorPtr();
                }
                boolean ret = predictWithOutput(modelParallelRunnerPtr, inputsPtrArray, outputsPtrArray);
                if (!ret) {
                    return false;
                }
                return true;
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
        } finally {
            rwLock.readLock().unlock();
        }
    }

    /**
     * Obtains all input tensors of the model.
     *
     * @return The vector that includes all input tensors.
     */
    public List<MSTensor> getInputs() {
        signal.incrementAndGet();
        rwLock.readLock().lock();
        if (this.modelParallelRunnerPtr == 0L) {
            rwLock.readLock().unlock();
            return Collections.emptyList();
        }
        List<Long> ret = this.getInputs(this.modelParallelRunnerPtr);
        List<MSTensor> tensors = new ArrayList<>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        rwLock.readLock().unlock();
        return tensors;
    }

    /**
     * Obtains all output tensors of the model.
     *
     * @return The vector that includes all input tensors.
     */
    public List<MSTensor> getOutputs() {
        signal.incrementAndGet();
        rwLock.readLock().lock();
        if (this.modelParallelRunnerPtr == 0L) {
            rwLock.readLock().unlock();
            return Collections.emptyList();
        }
        List<Long> ret = this.getOutputs(this.modelParallelRunnerPtr);
        List<MSTensor> tensors = new ArrayList<>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        rwLock.readLock().unlock();
        return tensors;
    }

    /**
     * Free model
     */
    public void free() {
        int maxCount = 100;
        while (maxCount > 0) {
            --maxCount;
            long preSignal = signal.get();
            rwLock.writeLock().lock();
            long curSignal = signal.get();
            rwLock.writeLock().unlock();
            if (curSignal != preSignal) {
                continue;
            }
            break;
        }
        rwLock.writeLock().lock();
        long modelParallelRunnerTempPtr = modelParallelRunnerPtr;
        modelParallelRunnerPtr = 0L;
        rwLock.writeLock().unlock();
        if (modelParallelRunnerTempPtr != 0L) {
            this.free(modelParallelRunnerTempPtr);
        }
    }

    private native long init(String modelPath, long runnerConfigPtr);

    private native List<Long> predict(long modelParallelRunnerPtr, long[] inputs);

    private native boolean predictWithOutput(long modelParallelRunnerPtr, long[] inputs, long[] outputs);

    private native List<Long> getInputs(long modelParallelRunnerPtr);

    private native List<Long> getOutputs(long modelParallelRunnerPtr);

    private native void free(long modelParallelRunnerPtr);
}
