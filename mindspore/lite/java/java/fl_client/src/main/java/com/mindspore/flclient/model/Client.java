/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

package com.mindspore.flclient.model;

import com.mindspore.Graph;
import com.mindspore.config.DeviceType;
import com.mindspore.config.MSContext;
import com.mindspore.config.TrainCfg;
import com.mindspore.flclient.Common;
import com.mindspore.flclient.LocalFLParameter;
import com.mindspore.flclient.common.FLLoggerGenerater;
import com.mindspore.MSTensor;
import com.mindspore.Model;
import mindspore.schema.FeatureMap;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * Defining the client base class.
 *
 * @since v1.0
 */
public abstract class Client {
    private static final Logger logger = FLLoggerGenerater.getModelLogger(Client.class.toString());

    /**
     * Mindspore model object.
     */
    public Model model;

    /**
     * dataset map.
     */
    public Map<RunType, DataSet> dataSets = new HashMap<>();

    private final List<ByteBuffer> inputsBuffer = new ArrayList<>();

    private float uploadLoss = 0.0f;
    /**
     * Get callback.
     *
     * @param runType dataset type.
     * @param dataSet dataset.
     * @return callback objects.
     */
    public abstract List<Callback> initCallbacks(RunType runType, DataSet dataSet);

    /**
     * Init datasets.
     *
     * @param files data files.
     * @return dataset sizes map.
     */
    public abstract Map<RunType, Integer> initDataSets(Map<RunType, List<String>> files);

    /**
     * Get eval accuracy.
     *
     * @param evalCallbacks callbacks for eval model.
     * @return eval accuracy.
     */
    public abstract float getEvalAccuracy(List<Callback> evalCallbacks);

    /**
     * Get infer model result.
     *
     * @param inferCallback callback used for infer model.
     * @return infer result.
     */
    public abstract List<Object> getInferResult(List<Callback> inferCallback);

    /**
     * Init lite session and inputs buffer.
     *
     * @param modelPath model file path.
     * @param config session run config.
     * @return execute status.
     */
    public Status initSessionAndInputs(String modelPath, int[][] inputShapes) {
        if (modelPath == null) {
            logger.severe("session init failed");
            return Status.FAILED;
        }
        if (!initSession(modelPath)) {
            free();
            return Status.FAILED;
        }
        inputsBuffer.clear();
        if (inputShapes == null) {
            List<MSTensor> inputs = model.getInputs();

            for (MSTensor input : inputs) {
                ByteBuffer inputBuffer = ByteBuffer.allocateDirect((int) input.size());
                inputBuffer.order(ByteOrder.nativeOrder());
                inputsBuffer.add(inputBuffer);
            }
        } else {
            boolean isSuccess = model.resize(model.getInputs(), inputShapes);
            if (!isSuccess) {
                logger.severe("session resize failed");
                return Status.FAILED;
            }
            for (int[] shapes : inputShapes) {
                int size = IntStream.of(shapes).reduce((a, b) -> a * b).getAsInt() * Integer.BYTES;
                ByteBuffer inputBuffer = ByteBuffer.allocateDirect(size);
                inputBuffer.order(ByteOrder.nativeOrder());
                inputsBuffer.add(inputBuffer);
            }
        }
        return Status.SUCCESS;
    }

    private void fillModelInput(DataSet dataSet, int batchIdx) {
        dataSet.fillInputBuffer(inputsBuffer, batchIdx);
        List<MSTensor> inputs = model.getInputs();
        for (int i = 0; i < inputs.size(); i++) {
            inputs.get(i).setData(inputsBuffer.get(i));
        }
    }

    /**
     * Train model.
     *
     * @param epochs train epochs.
     * @return execute status.
     */
    public Status trainModel(int epochs) {
        if (epochs <= 0) {
            logger.severe("epochs cannot smaller than 0");
            return Status.INVALID;
        }
        model.setTrainMode(true);
        DataSet trainDataSet = dataSets.getOrDefault(RunType.TRAINMODE, null);
        if (trainDataSet == null) {
            logger.severe("not find train dataset");
            return Status.NULLPTR;
        }
        trainDataSet.padding();
        List<Callback> trainCallbacks = initCallbacks(RunType.TRAINMODE, trainDataSet);
        Status status = runModel(epochs, trainCallbacks, trainDataSet);
        if (status != Status.SUCCESS) {
            logger.severe("train loop failed");
            return status;
        }
        return Status.SUCCESS;
    }

    /**
     * Eval model.
     *
     * @return eval accuracy.
     */
    public float evalModel() {
        model.setTrainMode(false);
        DataSet evalDataSet = dataSets.getOrDefault(RunType.EVALMODE, null);
        evalDataSet.padding();
        List<Callback> evalCallbacks = initCallbacks(RunType.EVALMODE, evalDataSet);
        Status status = runModel(1, evalCallbacks, evalDataSet);
        if (status != Status.SUCCESS) {
            logger.severe("train loop failed");
            return Float.NaN;
        }
        return getEvalAccuracy(evalCallbacks);
    }

    /**
     * Infer model.
     *
     * @return infer status.
     */
    public List<Object> inferModel() {
        model.setTrainMode(false);
        DataSet inferDataSet = dataSets.getOrDefault(RunType.INFERMODE, null);
        inferDataSet.padding();
        List<Callback> inferCallbacks = initCallbacks(RunType.INFERMODE, inferDataSet);
        Status status = runModel(1, inferCallbacks, inferDataSet);
        if (status != Status.SUCCESS) {
            logger.severe("train loop failed");
            return null;
        }
        return getInferResult(inferCallbacks);
    }

    private Status runModel(int epochs, List<Callback> callbacks, DataSet dataSet) {
        LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < dataSet.batchNum; j++) {
                if (localFLParameter.isStopJobFlag()) {
                    logger.info("the stopJObFlag is set to true, the job will be stop");
                    return Status.FAILED;
                }
                fillModelInput(dataSet, j);
                boolean isSuccess = model.runStep();
                if (!isSuccess) {
                    logger.severe("run graph failed");
                    return Status.FAILED;
                }
                for (Callback callBack : callbacks) {
                    callBack.stepEnd();
                }
            }
            for (Callback callBack : callbacks) {
                callBack.epochEnd();
                if (callBack instanceof LossCallback && i == epochs - 1) {
                    LossCallback lossCallback = (LossCallback)callBack;
                    setUploadLoss(lossCallback.getUploadLoss());
                }
            }
        }
        long endTime = System.currentTimeMillis();
        logger.info("total run time:" + (endTime - startTime) + "ms");
        return Status.SUCCESS;
    }

    /**
     * Save model.
     *
     * @param modelPath model file path.
     * @return save status.
     */
    public Status saveModel(String modelPath) {
        if (modelPath == null) {
            logger.severe("model path cannot be empty");
            return Status.NULLPTR;
        }
        boolean isSuccess = model.export(modelPath, 0, false, null);
        if (!isSuccess) {
            logger.severe("save model failed");
            return Status.FAILED;
        }
        return Status.SUCCESS;
    }

    private boolean initSession(String modelPath) {
        if (modelPath == null) {
            logger.severe("modelPath cannot be empty");
            return false;
        }
        int deviceType = LocalFLParameter.getInstance().getDeviceType();
        int threadNum = LocalFLParameter.getInstance().getThreadNum();
        int cpuBindMode = LocalFLParameter.getInstance().getCpuBindMode();
        boolean enableFp16 = LocalFLParameter.getInstance().isEnableFp16();
        MSContext msContext = new MSContext();
        // use default param init context
        if(!msContext.init(threadNum, cpuBindMode)){
            logger.severe("Call msContext.init failed, threadNum " + threadNum + ", cpuBindMode " + cpuBindMode);
            msContext.free();
            return false;
        }

        if (!msContext.addDeviceInfo(deviceType, enableFp16, 0)) {
            logger.severe("Call msContext.addDeviceInfo failed, deviceType " + deviceType + ", enableFp16 " + enableFp16);
            msContext.free();
            return false;
        }

        TrainCfg trainCfg = new TrainCfg();
        if(!trainCfg.init()){
            logger.severe("Call trainCfg.init failed ...");
            msContext.free();
            trainCfg.free();
            return false;
        }
        Graph graph = new Graph();
        if(!graph.load(modelPath)){
            logger.severe("Call graph.load failed, modelPath: " + modelPath);
            graph.free();
            trainCfg.free();
            msContext.free();
            return false;
        }
        model = new Model();
        if (!model.build(graph, msContext, trainCfg)) {
            // The Jni implement will change msContext & graph to shared_ptr, no need free here
            logger.severe("Call model.build failed ... ");
            graph.free();
            return false;
        }
        graph.free();
        return true;
    }

    /**
     * Get model feature maps.
     *
     * @return model weights.
     */
    public List<MSTensor> getFeatures() {
        if (model == null) {
            return new ArrayList<>();
        }
        return model.getFeatureMaps();
    }

    /**
     * update model feature maps.
     *
     * @param modelName model file name.
     * @param featureMaps new weights.
     * @return update status.
     */
    public Status updateFeatures(String modelName, List<FeatureMap> featureMaps) {
        if (model == null || featureMaps == null || modelName == null || modelName.isEmpty()) {
            logger.severe("trainSession,featureMaps modelName cannot be null");
            return Status.NULLPTR;
        }

        List<MSTensor> modelFeatures = model.getFeatureMaps();
        HashMap<String, MSTensor> modelFeatureMaps = new HashMap<String, MSTensor>();
        for (MSTensor item : modelFeatures) {
            modelFeatureMaps.put(item.tensorName(), item);
        }

        List<MSTensor> tensors = new ArrayList<>(featureMaps.size());
        for (FeatureMap newFeature : featureMaps) {
            if (newFeature == null) {
                logger.severe("newFeature cannot be null");
                return Status.NULLPTR;
            }

            if (newFeature.weightFullname().isEmpty() || !modelFeatureMaps.containsKey(newFeature.weightFullname())) {
                logger.severe("Can't get feature for name:" + newFeature.weightFullname());
                return Status.NULLPTR;
            }
            MSTensor origin = modelFeatureMaps.get(newFeature.weightFullname());
            int dataType = origin.getDataType();
            int[] dataShape = origin.getShape();

            ByteBuffer by = newFeature.dataAsByteBuffer();
            ByteBuffer newData = ByteBuffer.allocateDirect(by.remaining());
            newData.order(ByteOrder.nativeOrder());
            newData.put(by);
            MSTensor tensor = MSTensor.createTensor(newFeature.weightFullname(), dataType, dataShape, newData);
            tensors.add(tensor);
        }
        boolean isSuccess = model.updateFeatureMaps(tensors);
        for (MSTensor tensor : tensors) {
            if (tensor == null) {
                logger.severe("tensor cannot be null");
                return Status.NULLPTR;
            }
            tensor.free();
        }

        if (isSuccess) {
            model.export(modelName, 0, false, null);
            return Status.SUCCESS;
        }
        return Status.FAILED;
    }

    /**
     * Free client.
     */
    public void free() {
        if (model != null) {
            model.free();
            model = null;
        }
    }

    /**
     * Set learning rate.
     *
     * @param lr learning rate.
     * @return execute status.
     */
    public Status setLearningRate(float lr) {
        if (model.setLearningRate(lr)) {
            return Status.SUCCESS;
        }
        logger.severe("set learning rate failed");
        return Status.FAILED;
    }

    /**
     * Set client batch size.
     *
     * @param batchSize batch size.
     */
    public void setBatchSize(int batchSize) {
        for (DataSet dataset : dataSets.values()) {
            dataset.batchSize = batchSize;
        }
    }

    public float getUploadLoss() {
        return uploadLoss;
    }

    public void setUploadLoss(float uploadLoss) {
        this.uploadLoss = uploadLoss;
    }
}