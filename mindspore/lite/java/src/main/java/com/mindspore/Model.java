/*
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

import static com.mindspore.config.MindsporeLite.POINTER_DEFAULT_VALUE;

import com.mindspore.config.MSContext;
import com.mindspore.config.DataType;
import com.mindspore.config.MindsporeLite;
import com.mindspore.config.TrainCfg;

import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Logger;
import java.lang.reflect.Array;

/**
 * The Model class is used to define a MindSpore model, facilitating computational graph management.
 *
 * @since v1.0
 */
public class Model {
    private static final Logger LOGGER = Logger.getLogger(Model.class.toString());

    static {
        MindsporeLite.init();
    }

    private long modelPtr = POINTER_DEFAULT_VALUE;
    private boolean isModelSharePtr = false;
    private List<MSTensor> inputTensors = null;

    /**
     * Construct function.
     */
    public Model() {
        this.isModelSharePtr = false;
        this.modelPtr = this.createModel();
        this.inputTensors = null;
    }

    /**
     * Construct function.
     *
     * @param modelPtr model shared pointer.
     */
    public Model(long modelPtr) {
        this.isModelSharePtr = true;
        this.modelPtr = modelPtr;
        this.inputTensors = null;
    }

    /**
     * Build model by graph.
     *
     * @param graph   graph contains the buffer.
     * @param context model build context.
     * @param cfg     model build train config.used for train.
     * @return build status.
     */
    public boolean build(Graph graph, MSContext context, TrainCfg cfg) {
        if (graph == null || context == null) {
            return false;
        }
        long cfgPtr = cfg != null ? cfg.getTrainCfgPtr() : POINTER_DEFAULT_VALUE;
        return this.buildByGraph(modelPtr, graph.getGraphPtr(), context.getMSContextPtr(), cfgPtr);
    }

    /**
     * Build model.
     *
     * @param buffer          model buffer.
     * @param modelType       model type.
     * @param context         model build context.
     * @param decKey         define the key used to decrypt the ciphertext model. The key length is 16.
     * @param decMode        define the decryption mode. Options: AES-GCM.
     * @param croptoLibPath   define the openssl library path.
     * @return model build status.
     */
    public boolean build(final MappedByteBuffer buffer, int modelType, MSContext context, char[] decKey, String decMode,
                         String croptoLibPath) {
        boolean isValid = (context != null && buffer != null && decKey != null && decMode != null &&
                           croptoLibPath != null);
        if (!isValid) {
            return false;
        }
        return this.buildByBuffer(modelPtr, buffer, modelType, context.getMSContextPtr(), decKey, decMode,
                                  croptoLibPath);
    }

    /**
     * Build model.
     *
     * @param buffer    model buffer.
     * @param modelType model type.
     * @param context   model build context.
     * @return model build status.
     */
    public boolean build(final MappedByteBuffer buffer, int modelType, MSContext context) {
        if (context == null || buffer == null) {
            return false;
        }
        return this.buildByBuffer(modelPtr, buffer, modelType, context.getMSContextPtr(), null, "", "");
    }


    /**
     * Build model.
     *
     * @param modelPath       model path.
     * @param modelType       model type.
     * @param context         model build context.
     * @param decKey          define the key used to decrypt the ciphertext model. The key length is 16.
     * @param decMode         define the decryption mode. Options: AES-GCM.
     * @param croptoLibPath   define the openssl library path.
     * @return model build status.
     */
    public boolean build(String modelPath, int modelType, MSContext context, char[] decKey, String decMode,
                         String croptoLibPath) {
        boolean isValid = (context != null && modelPath != null && decKey != null && decMode != null &&
                                   croptoLibPath != null);
        if (!isValid) {
            return false;
        }
        return this.buildByPath(modelPtr, modelPath, modelType, context.getMSContextPtr(), decKey, decMode,
                                croptoLibPath);
    }

    /**
     * Build model.
     *
     * @param modelPath model path.
     * @param modelType model type.
     * @param context   model build context.
     * @return build status.
     */
    public boolean build(String modelPath, int modelType, MSContext context) {
        if (context == null || modelPath == null) {
            return false;
        }
        return this.buildByPath(modelPtr, modelPath, modelType, context.getMSContextPtr(), null, "", "");
    }

    /**
     * Execute predict.
     *
     * @return predict status.
     */
    public boolean predict() {
        List<MSTensor> inputs = this.getInputs();
        if (inputs == null || inputs.size() == 0) {
            return false;
        }
        long[] inputsPtrArray = new long[inputs.size()];
        Object[] bufferArray = new Object[inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            inputsPtrArray[i] = inputs.get(i).getMSTensorPtr();
            Object obj = inputs.get(i).getData();
            if (Array.getLength(obj) == 0) {
                return false;
            }
            bufferArray[i] = obj;
        }
        return this.runStep(modelPtr, inputsPtrArray, bufferArray);
    }

    /**
     * Run Model by step.
     *
     * @return run model status.work in train mode.
     */
    public boolean runStep() {
        return this.predict();
    }

    /**
     * Resize inputs shape.
     *
     * @param inputs Model inputs.
     * @param dims   Define the new inputs shape.
     * @return Whether the resize is successful.
     */
    public boolean resize(List<MSTensor> inputs, int[][] dims) {
        if (inputs == null || dims == null) {
            return false;
        }
        long[] inputsArray = new long[inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            inputsArray[i] = inputs.get(i).getMSTensorPtr();
        }
        return this.resize(this.modelPtr, inputsArray, dims);
    }

    /**
     * Get model inputs tensor.
     *
     * @return input tensors.
     */
    public List<MSTensor> getInputs() {
        if (this.inputTensors != null){
            return this.inputTensors;
        }
        List<Long> tensorAddrs = this.getInputs(this.modelPtr);
        this.inputTensors = new ArrayList<>(tensorAddrs.size());
        for (Long msTensorAddr : tensorAddrs) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            this.inputTensors.add(msTensor);
        }
        return this.inputTensors;
    }

    /**
     * Get model outputs.
     *
     * @return model outputs tensor.
     */
    public List<MSTensor> getOutputs() {
        List<Long> tensorAddrs = this.getOutputs(this.modelPtr);
        List<MSTensor> tensors = new ArrayList<>(tensorAddrs.size());
        for (Long msTensorAddr : tensorAddrs) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Get input tensor by tensor name.
     *
     * @param tensorName name.
     * @return input tensor.
     */
    public MSTensor getInputByTensorName(String tensorName) {
        List<MSTensor> inputTensors = this.getInputs();
        for (int i = 0; i < inputTensors.size(); i++) {
            MSTensor tensor = inputTensors.get(i);
            if (tensor.tensorName().equals(tensorName)) {
                return tensor;
            }
        }
        return null;
    }

    /**
     * Get output tensor by tensor name.
     *
     * @param tensorName output tensor name
     * @return output tensor
     */
    public MSTensor getOutputByTensorName(String tensorName) {
        if (tensorName == null) {
            return null;
        }
        long tensorAddr = this.getOutputByTensorName(this.modelPtr, tensorName);
        if (tensorAddr == POINTER_DEFAULT_VALUE) {
            return null;
        }
        return new MSTensor(tensorAddr);
    }

    /**
     * Get output tensors by node name.
     *
     * @param nodeName output node name
     * @return output tensor
     */
    public List<MSTensor> getOutputsByNodeName(String nodeName) {
        if (nodeName == null) {
            return new ArrayList<>();
        }
        List<Long> tensorAddrs = this.getOutputsByNodeName(this.modelPtr, nodeName);
        List<MSTensor> tensors = new ArrayList<>(tensorAddrs.size());
        for (Long msTensorAddr : tensorAddrs) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Get output tensor names.
     *
     * @return output tensor name list.
     */
    public List<String> getOutputTensorNames() {
        return this.getOutputTensorNames(this.modelPtr);
    }

    /**
     * Load config file.
     *
     * @param configPath          config file path.
     *
     * @return Whether the LoadConfig is successful.
     */
    public boolean loadConfig(String configPath) {
        return loadConfig(modelPtr, configPath);
    }

    /**
     * Update config.
     *
     * @param section define the config section.
     * @param config define the config will be updated.
     *
     * @return Whether the updateConfig is successful.
     */
    public boolean updateConfig(String section, HashMap<String, String> config) {
        return updateConfig(modelPtr, section, config);
    }

    /**
     * Export the model.
     *
     * @param fileName          Name Model file name.
     * @param quantizationType  The quant type.0,no_quant,1,weight_quant,2,full_quant.
     * @param isOnlyExportInfer if export only inferece.
     * @param outputTensorNames tensor name used for export inference graph.
     * @return Whether the export is successful.
     */
    public boolean export(String fileName, int quantizationType, boolean isOnlyExportInfer, List<String> outputTensorNames) {
        if (fileName == null) {
            return false;
        }
        if (outputTensorNames != null) {
            String[] outputTensorArray = new String[outputTensorNames.size()];
            for (int i = 0; i < outputTensorNames.size(); i++) {
                outputTensorArray[i] = outputTensorNames.get(i);
            }
            return export(modelPtr, fileName, quantizationType, isOnlyExportInfer, outputTensorArray);
        }
        return export(modelPtr, fileName, quantizationType, isOnlyExportInfer, null);
    }

    /**
     * Export model's weights, which can be used in micro only.
     *
     * @param weightFile                  The path of exported weight file.
     * @param isInference                 Whether to export weights from a reasoning model. Currently, only support`true`.
     * @param enableFp16                  Float-weight is whether to be saved in float16 format.
     * @param changeableWeightNames       The set the name of these weight tensors, whose shape is changeable.
     * @return
     */
    public boolean exportWeightsCollaborateWithMicro(String weightFile, boolean isInference,
                                 boolean enableFp16, List<String> changeableWeightNames) {
        if (weightFile == null || weightFile.length() == 0) {
            LOGGER.severe("Input params invalid.");
            return false;
        }
        return exportWeightsCollaborateWithMicro(modelPtr, weightFile, isInference, enableFp16,
                             changeableWeightNames.toArray(new String[0]));
    }

    /**
     * Get the FeatureMap.
     *
     * @return FeaturesMap Tensor list.
     */
    public List<MSTensor> getFeatureMaps() {
        List<Long> tensorAddrs = this.getFeatureMaps(this.modelPtr);
        ArrayList<MSTensor> tensors = new ArrayList<>(tensorAddrs.size());
        for (Long msTensorAddr : tensorAddrs) {
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
    public boolean updateFeatureMaps(List<MSTensor> features) {
        if (features == null) {
            return false;
        }
        long[] inputsArray = new long[features.size()];
        for (int i = 0; i < features.size(); i++) {
            inputsArray[i] = features.get(i).getMSTensorPtr();
        }
        return this.updateFeatureMaps(modelPtr, inputsArray);
    }

    /**
     * Set model work train mode
     *
     * @param isTrain is train mode.true work train mode.
     * @return set status.
     */
    public boolean setTrainMode(boolean isTrain) {
        return this.setTrainMode(modelPtr, isTrain);
    }

    /**
     * Get train mode
     *
     * @return train mode.
     */
    public boolean getTrainMode() {
        return this.getTrainMode(modelPtr);
    }

    /**
     * set learning rate.
     *
     * @param learning_rate learning rate.
     * @return Whether the set learning rate is successful.
     */
    public boolean setLearningRate(float learning_rate) {
        return this.setLearningRate(this.modelPtr, learning_rate);
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
        return this.setupVirtualBatch(this.modelPtr, virtualBatchMultiplier, learningRate, momentum);
    }

    /**
     * Free model
     */
    public void free() {
        if (this.inputTensors != null){
            for (MSTensor tensor : this.inputTensors) {
                tensor.free();
            }
            this.inputTensors = null;
        }
        this.free(modelPtr, isModelSharePtr);
    }

    private native long createModel();

    private native void free(long modelPtr, boolean isShared);

    private native boolean buildByGraph(long modelPtr, long graphPtr, long contextPtr, long cfgPtr);

    private native boolean buildByPath(long modelPtr, String modelPath, int modelType, long contextPtr,
                                    char[] dec_key, String dec_mod, String cropto_lib_path);

    private native boolean buildByBuffer(long modelPtr, MappedByteBuffer buffer, int modelType, long contextPtr,
                                      char[] dec_key, String dec_mod, String cropto_lib_path);

    private native List<Long> getInputs(long modelPtr);

    private native long getInputByTensorName(long modelPtr, String tensorName);

    private native boolean runStep(long modelPtr, long[] inputs, Object[] buffer);

    private native List<Long> getOutputs(long modelPtr);

    private native long getOutputByTensorName(long modelPtr, String tensorName);

    private native List<String> getOutputTensorNames(long modelPtr);

    private native List<Long> getOutputsByNodeName(long modelPtr, String nodeName);

    private native boolean setTrainMode(long modelPtr, boolean isTrain);

    private native boolean getTrainMode(long modelPtr);

    private native boolean resize(long modelPtr, long[] inputs, int[][] dims);

    private native boolean loadConfig(long modelPtr, String configPath);

    private native boolean updateConfig(long modelPtr, String section, HashMap<String, String> config);

    private native boolean export(long modelPtr, String fileName, int quantizationType, boolean isOnlyExportInfer, String[] outputTensorNames);

    private native boolean exportWeightsCollaborateWithMicro(long modelPtr, String weightFile, boolean isInference,
                                         boolean enableFp16, String[] changeableWeightNames);

    private native List<Long> getFeatureMaps(long modelPtr);

    private native boolean updateFeatureMaps(long modelPtr, long[] newFeatures);

    private native boolean setLearningRate(long modelPtr, float learning_rate);

    private native boolean setupVirtualBatch(long modelPtr, int virtualBatchMultiplier, float learningRate, float momentum);
}
