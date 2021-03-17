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

package com.mindspore.lite;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.mindspore.lite.config.MSConfig;

public class TrainSession {
    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private long sessionPtr;

    public TrainSession() {
        this.sessionPtr = 0;
    }

    public boolean init(String modelFilename, MSConfig config) {
        this.sessionPtr = createSession(modelFilename, config.getMSConfigPtr());
        return this.sessionPtr != 0;
    }

    public long getSessionPtr() {
        return sessionPtr;
    }

    public void bindThread(boolean if_bind) {
        this.bindThread(this.sessionPtr, if_bind);
    }

    public boolean runGraph() {
        return this.runGraph(this.sessionPtr);
    }

    public List<MSTensor> getInputs() {
        List<Long> ret = this.getInputs(this.sessionPtr);
        ArrayList<MSTensor> tensors = new ArrayList<MSTensor>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    public MSTensor getInputsByTensorName(String tensorName) {
        Long tensorAddr = this.getInputsByTensorName(this.sessionPtr, tensorName);
        if (tensorAddr == null) {
            return null;
        }
        MSTensor msTensor = new MSTensor(tensorAddr);
        return msTensor;
    }

    public List<MSTensor> getOutputsByNodeName(String nodeName) {
        List<Long> ret = this.getOutputsByNodeName(this.sessionPtr, nodeName);
        ArrayList<MSTensor> tensors = new ArrayList<>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    public Map<String, MSTensor> getOutputMapByTensor() {
        Map<String, Long> ret = this.getOutputMapByTensor(this.sessionPtr);
        Map<String, MSTensor> tensorMap = new HashMap<>();
        Set<Map.Entry<String, Long>> entrySet = ret.entrySet();
        for (Map.Entry<String, Long> entry : entrySet) {
            String name = entry.getKey();
            Long msTensorAddr = entry.getValue();
            tensorMap.put(name, new MSTensor(msTensorAddr));
        }
        return tensorMap;
    }

    public List<String> getOutputTensorNames() {
        return getOutputTensorNames(this.sessionPtr);
    }

    public MSTensor getOutputByTensorName(String tensorName) {
        Long tensorAddr = getOutputByTensorName(this.sessionPtr, tensorName);
        if (tensorAddr == null) {
            return null;
        }
        return new MSTensor(tensorAddr);
    }

    public void free() {
        this.free(this.sessionPtr);
        this.sessionPtr = 0;
    }

    public boolean resize(List<MSTensor> inputs, int[][] dims) {
        long[] inputsArray = new long[inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            inputsArray[i] = inputs.get(i).getMSTensorPtr();
        }
        return this.resize(this.sessionPtr, inputsArray, dims);
    }

    public boolean saveToFile(String modelFilename) {
        return this.saveToFile(this.sessionPtr, modelFilename);
    }

    public boolean train() {
        return this.train(this.sessionPtr);
    }

    public boolean eval() {
        return this.eval(this.sessionPtr);
    }

    public boolean isTrain() {
        return this.isTrain(this.sessionPtr);
    }

    public boolean isEval() {
        return this.isEval(this.sessionPtr);
    }

    public boolean setLearningRate(float learning_rate) {
        return this.setLearningRate(this.sessionPtr, learning_rate);
    }

    public boolean setupVirtualBatch(int virtualBatchMultiplier, float learningRate, float momentum) {
        return this.setupVirtualBatch(this.sessionPtr, virtualBatchMultiplier, learningRate, momentum);
    }

    public boolean setupVirtualBatch(int virtualBatchMultiplier) {
        return this.setupVirtualBatch(this.sessionPtr, virtualBatchMultiplier, -1.0f, -1.0f);
    }

    public boolean setLossName(String lossName) {
        return this.setLossName(this.sessionPtr,lossName);
    }
    
    
    private native long createSession(String modelFilename, long msConfigPtr);

    private native void bindThread(long sessionPtr, boolean if_bind);

    private native boolean runGraph(long sessionPtr);

    private native List<Long> getInputs(long sessionPtr);

    private native long getInputsByTensorName(long sessionPtr, String tensorName);

    private native List<Long> getOutputsByNodeName(long sessionPtr, String nodeName);

    private native Map<String, Long> getOutputMapByTensor(long sessionPtr);

    private native List<String> getOutputTensorNames(long sessionPtr);

    private native long getOutputByTensorName(long sessionPtr, String tensorName);

    private native void free(long sessionPtr);

    private native boolean resize(long sessionPtr, long[] inputs, int[][] dims);

    private native boolean saveToFile(long sessionPtr, String modelFilename);

    private native boolean train(long sessionPtr);

    private native boolean eval(long sessionPtr);

    private native boolean isTrain(long sessionPtr);

    private native boolean isEval(long sessionPtr);

    private native boolean setLearningRate(long sessionPtr, float learning_rate);

    private native boolean setupVirtualBatch(long sessionPtr, int virtualBatchMultiplier, float learningRate, float momentum);

    private native boolean setLossName(long sessionPtr,String lossName);
}
