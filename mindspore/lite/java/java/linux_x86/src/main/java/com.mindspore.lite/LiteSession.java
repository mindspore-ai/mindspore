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

package com.mindspore.lite;

import com.mindspore.lite.config.MSConfig;

import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * LiteSession defines session in MindSpore Lite for compiling Model and forwarding model.
 *
 * @since v1.0
 */
public class LiteSession {
    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private long sessionPtr = 0;

    /**
     * LiteSession construct
     * Deprecated, please use "public static LiteSession createSession(final MSConfig config)" instead.
     */
    public LiteSession() {
        this.sessionPtr = 0;
    }

    /**
     * Initialize LiteSession.
     * Deprecated, please use "public static LiteSession createSession(final MSConfig config)" instead.
     *
     * @param config MSconfig
     * @return Whether the initialization is successful.
     */
    public boolean init(MSConfig config) {
        this.sessionPtr = createSession(config.getMSConfigPtr());
        return this.sessionPtr != 0;
    }

    /**
     * Use MSConfig to create Litessesion
     *
     * @param config Msconfig
     * @return LiteSession Object
     */
    public static LiteSession createSession(final MSConfig config) {
        LiteSession liteSession = new LiteSession();
        liteSession.sessionPtr = liteSession.createSession(config.getMSConfigPtr());
        if (liteSession.sessionPtr == 0) {
            return null;
        } else {
            return liteSession;
        }
    }

    /**
     * Use Model buffer and MSConfig to create Litessesion
     *
     * @param buffer model buffer
     * @param config MSConfig
     * @return LiteSession Object
     */
    public static LiteSession createSession(final MappedByteBuffer buffer, final MSConfig config) {
        LiteSession liteSession = new LiteSession();
        liteSession.sessionPtr = liteSession.createSessionWithModel(buffer, config.getMSConfigPtr());
        if (liteSession.sessionPtr == 0) {
            return null;
        } else {
            return liteSession;
        }
    }

    /**
     * Get Session pointer
     *
     * @return Session pointer
     */
    public long getSessionPtr() {
        return sessionPtr;
    }

    /**
     * Set Session pointer
     *
     * @param sessionPtr set Session pointer
     */
    public void setSessionPtr(long sessionPtr) {
        this.sessionPtr = sessionPtr;
    }

    /**
     * Attempt to bind or unbind threads in the thread pool to or from the specified cpu core.
     *
     * @param isBind Define whether to bind or unbind threads.
     */
    public void bindThread(boolean isBind) {
        this.bindThread(this.sessionPtr, isBind);
    }

    /**
     * Compile MindSpore Lite model.
     *
     * @param model Define the model to be compiled.
     * @return Whether the compilation is successful.
     */
    public boolean compileGraph(Model model) {
        return this.compileGraph(this.sessionPtr, model.getModelPtr());
    }

    /**
     * Run the session for inference.
     *
     * @return Whether the inference is successful.
     */
    public boolean runGraph() {
        return this.runGraph(this.sessionPtr);
    }

    /**
     * Get the MSTensors input of MindSpore Lite model.
     *
     * @return The vector of MindSpore Lite MSTensor.
     */
    public List<MSTensor> getInputs() {
        List<Long> ret = this.getInputs(this.sessionPtr);
        ArrayList<MSTensor> tensors = new ArrayList<MSTensor>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Get the MSTensors input of MindSpore Lite model by the node name.
     *
     * @param tensorName Define the tensor name.
     * @return MindSpore Lite MSTensor.
     */
    public MSTensor getInputsByTensorName(String tensorName) {
        Long tensorAddr = this.getInputsByTensorName(this.sessionPtr, tensorName);
        if (tensorAddr == null) {
            return null;
        }
        return new MSTensor(tensorAddr);
    }

    /**
     * Get the MSTensors output of MindSpore Lite model by the node name.
     *
     * @param nodeName Define the node name.
     * @return The vector of MindSpore Lite MSTensor.
     */
    public List<MSTensor> getOutputsByNodeName(String nodeName) {
        List<Long> ret = this.getOutputsByNodeName(this.sessionPtr, nodeName);
        ArrayList<MSTensor> tensors = new ArrayList<>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Get the MSTensors output of the MindSpore Lite model associated with the tensor name.
     *
     * @return The map of output tensor name and MindSpore Lite MSTensor.
     */
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

    /**
     * Get the name of output tensors of the model compiled by this session.
     *
     * @return The vector of string as output tensor names in order.
     */
    public List<String> getOutputTensorNames() {
        return getOutputTensorNames(this.sessionPtr);
    }

    /**
     * Get the MSTensors output of MindSpore Lite model by the tensor name.
     *
     * @param tensorName Define the tensor name.
     * @return Pointer of MindSpore Lite MSTensor.
     */
    public MSTensor getOutputByTensorName(String tensorName) {
        Long tensorAddr = getOutputByTensorName(this.sessionPtr, tensorName);
        if (tensorAddr == null) {
            return null;
        }
        return new MSTensor(tensorAddr);
    }

    /**
     * Free LiteSession.
     */
    public void free() {
        this.free(this.sessionPtr);
        this.sessionPtr = 0;
    }

    /**
     * Resize inputs shape.
     *
     * @param inputs Model inputs.
     * @param dims   Define the new inputs shape.
     * @return Whether the resize is successful.
     */
    public boolean resize(List<MSTensor> inputs, int[][] dims) {
        long[] inputsArray = new long[inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            inputsArray[i] = inputs.get(i).getMSTensorPtr();
        }
        return this.resize(this.sessionPtr, inputsArray, dims);
    }

    /**
     * Export the model.
     *
     * @param modelFile        Name Model file name.
     * @param modelType        Train or Inference type.
     * @param quantizationType The quant type.
     * @return Whether the export is successful.
     */
    public boolean export(String modelFileName, int modelType, int quantizationType) {
        return this.export(this.sessionPtr, modelFileName, modelType, quantizationType);
    }

    /**
     * Switch to the train mode
     *
     * @return Whether switch to the train mode is successful.
     */
    public boolean train() {
        return this.train(this.sessionPtr);
    }

    /**
     * Switch to the eval mode
     *
     * @return Whether switch to the eval mode is successful.
     */
    public boolean eval() {
        return this.eval(this.sessionPtr);
    }

    /**
     * Whether is train mode.
     *
     * @return Whether is train mode.
     */
    public boolean isTrain() {
        return this.isTrain(this.sessionPtr);
    }

    /**
     * Whether is eval mode.
     *
     * @return Whether is eval mode.
     */
    public boolean isEval() {
        return this.isEval(this.sessionPtr);
    }

    /**
     * set learning rate.
     *
     * @param learning_rate learning rate.
     * @return Whether the set learning rate is successful.
     */
    public boolean setLearningRate(float learning_rate) {
        return this.setLearningRate(this.sessionPtr, learning_rate);
    }

    /**
     * Set the virtual batch.
     *
     * @param virtualBatchMultiplier virtual batch multuplier.
     * @param learningRate           learning rate.
     * @param momentum               monentum.
     * @return Whether the virtual batch is successfully set.
     */
    public boolean setupVirtualBatch(int virtualBatchMultiplier, float learningRate, float momentum) {
        return this.setupVirtualBatch(this.sessionPtr, virtualBatchMultiplier, learningRate, momentum);
    }

    /**
     * Get the FeatureMap.
     *
     * @return FeaturesMap Tensor list.
     */
    public List<MSTensor> getFeaturesMap() {
        List<Long> ret = this.getFeaturesMap(this.sessionPtr);
        ArrayList<MSTensor> tensors = new ArrayList<MSTensor>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Update model Features.
     *
     * @param features new FeatureMap Tensor List.
     * @return Whether the model features is successfully update.
     */
    public boolean updateFeatures(List<MSTensor> features) {
        long[] inputsArray = new long[features.size()];
        for (int i = 0; i < features.size(); i++) {
            inputsArray[i] = features.get(i).getMSTensorPtr();
        }
        return this.updateFeatures(this.sessionPtr, inputsArray);
    }

    private native long createSession(long msConfigPtr);

    private native long createSessionWithModel(MappedByteBuffer buffer, long msConfigPtr);

    private native boolean compileGraph(long sessionPtr, long modelPtr);

    private native void bindThread(long sessionPtr, boolean isBind);

    private native boolean runGraph(long sessionPtr);

    private native List<Long> getInputs(long sessionPtr);

    private native long getInputsByTensorName(long sessionPtr, String tensorName);

    private native List<Long> getOutputsByNodeName(long sessionPtr, String nodeName);

    private native Map<String, Long> getOutputMapByTensor(long sessionPtr);

    private native List<String> getOutputTensorNames(long sessionPtr);

    private native long getOutputByTensorName(long sessionPtr, String tensorName);

    private native void free(long sessionPtr);

    private native boolean resize(long sessionPtr, long[] inputs, int[][] dims);

    private native boolean export(long sessionPtr, String modelFileName, int modelType, int quantizationType);

    private native boolean train(long sessionPtr);

    private native boolean eval(long sessionPtr);

    private native boolean isTrain(long sessionPtr);

    private native boolean isEval(long sessionPtr);

    private native boolean setLearningRate(long sessionPtr, float learningRate);

    private native boolean setupVirtualBatch(long sessionPtr, int virtualBatchMultiplier, float learningRate, float momentum);

    private native boolean updateFeatures(long sessionPtr, long[] newFeatures);

    private native List<Long> getFeaturesMap(long sessionPtr);
}
