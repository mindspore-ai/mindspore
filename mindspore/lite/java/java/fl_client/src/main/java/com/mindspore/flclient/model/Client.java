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

import com.mindspore.flclient.Common;
import com.mindspore.flclient.LocalFLParameter;
import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.Model;
import com.mindspore.lite.TrainSession;
import com.mindspore.lite.config.MSConfig;
import mindspore.schema.FeatureMap;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * Defining the client base class.
 *
 * @since v1.0
 */
public abstract class Client {
    private static final Logger logger = Logger.getLogger(Client.class.toString());

    /**
     * lite session object.
     */
    public LiteSession trainSession;

    /**
     * dataset map.
     */
    public Map<RunType, DataSet> dataSets = new HashMap<>();

    private final List<ByteBuffer> inputsBuffer = new ArrayList<>();

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
    public Status initSessionAndInputs(String modelPath, MSConfig config, int[][] inputShapes) {
        if (modelPath == null) {
            logger.severe(Common.addTag("session init failed"));
            return Status.FAILED;
        }
        Optional<LiteSession> optTrainSession = initSession(modelPath, config, inputShapes != null);
        if (!optTrainSession.isPresent()) {
            logger.severe(Common.addTag("session init failed"));
            return Status.FAILED;
        }
        trainSession = optTrainSession.get();
        inputsBuffer.clear();
        if (inputShapes == null) {
            List<MSTensor> inputs = trainSession.getInputs();
            for (MSTensor input : inputs) {
                ByteBuffer inputBuffer = ByteBuffer.allocateDirect((int) input.size());
                inputBuffer.order(ByteOrder.nativeOrder());
                inputsBuffer.add(inputBuffer);
            }
        } else {
            boolean isSuccess = trainSession.resize(trainSession.getInputs(), inputShapes);
            if (!isSuccess) {
                logger.severe(Common.addTag("session resize failed"));
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
        List<MSTensor> inputs = trainSession.getInputs();
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
        boolean isSuccess = trainSession.train();
        if (!isSuccess) {
            logger.severe(Common.addTag("train session switch eval mode failed"));
            return Status.FAILED;
        }
        if (epochs <= 0) {
            logger.severe(Common.addTag("epochs cannot smaller than 0"));
            return Status.INVALID;
        }

        DataSet trainDataSet = dataSets.getOrDefault(RunType.TRAINMODE, null);
        if (trainDataSet == null) {
            logger.severe(Common.addTag("not find train dataset"));
            return Status.NULLPTR;
        }
        trainDataSet.padding();
        List<Callback> trainCallbacks = initCallbacks(RunType.TRAINMODE, trainDataSet);
        Status status = runModel(epochs, trainCallbacks, trainDataSet);
        if (status != Status.SUCCESS) {
            logger.severe(Common.addTag("train loop failed"));
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
        boolean isSuccess = trainSession.eval();
        if (!isSuccess) {
            logger.severe(Common.addTag("train session switch eval mode failed"));
            return Float.NaN;
        }
        DataSet evalDataSet = dataSets.getOrDefault(RunType.EVALMODE, null);
        evalDataSet.padding();
        List<Callback> evalCallbacks = initCallbacks(RunType.EVALMODE, evalDataSet);
        Status status = runModel(1, evalCallbacks, evalDataSet);
        if (status != Status.SUCCESS) {
            logger.severe(Common.addTag("train loop failed"));
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
        boolean isSuccess = trainSession.eval();
        if (!isSuccess) {
            logger.severe(Common.addTag("train session switch eval mode failed"));
            return null;
        }
        DataSet inferDataSet = dataSets.getOrDefault(RunType.INFERMODE, null);
        inferDataSet.padding();
        List<Callback> inferCallbacks = initCallbacks(RunType.INFERMODE, inferDataSet);
        Status status = runModel(1, inferCallbacks, inferDataSet);
        if (status != Status.SUCCESS) {
            logger.severe(Common.addTag("train loop failed"));
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
                    logger.info(Common.addTag("the stopJObFlag is set to true, the job will be stop"));
                    return Status.FAILED;
                }
                fillModelInput(dataSet, j);
                boolean isSuccess = trainSession.runGraph();
                if (!isSuccess) {
                    logger.severe(Common.addTag("run graph failed"));
                    return Status.FAILED;
                }
                for (Callback callBack : callbacks) {
                    callBack.stepEnd();
                }
            }
            for (Callback callBack : callbacks) {
                callBack.epochEnd();
            }
        }
        long endTime = System.currentTimeMillis();
        logger.info(Common.addTag("total run time:" + (endTime - startTime) + "ms"));
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
            logger.severe(Common.addTag("model path cannot be empty"));
            return Status.NULLPTR;
        }
        boolean isSuccess = trainSession.export(modelPath, 0, 1);
        if (!isSuccess) {
            logger.severe(Common.addTag("save model failed"));
            return Status.FAILED;
        }
        return Status.SUCCESS;
    }

    private Optional<LiteSession> initSession(String modelPath, MSConfig msConfig, boolean isDynamicInferModel) {
        if (modelPath == null) {
            logger.severe(Common.addTag("modelPath cannot be empty"));
            return Optional.empty();
        }
        // only lite session support dynamic shape
        if (isDynamicInferModel) {
            Model model = new Model();
            boolean isSuccess = model.loadModel(modelPath);
            if (!isSuccess) {
                logger.severe(Common.addTag("load model failed:" + modelPath));
                return Optional.empty();
            }
            trainSession = LiteSession.createSession(msConfig);
            if (trainSession == null) {
                logger.severe(Common.addTag("init session failed,please check model :" + modelPath + " is valid or " +
                        "disk space is enough"));
                msConfig.free();
                model.free();
                return Optional.empty();
            }
            msConfig.free();
            isSuccess = trainSession.compileGraph(model);
            if (!isSuccess) {
                logger.severe(Common.addTag("compile graph failed:" + modelPath));
                model.free();
                trainSession.free();
                return Optional.empty();
            }
            model.free();
            return Optional.of(trainSession);
        } else {
            trainSession = TrainSession.createTrainSession(modelPath, msConfig, false);
            if (trainSession == null) {
                logger.severe(Common.addTag("init session failed,please check model :" + modelPath + " is valid or " +
                        "disk space is enough"));
                return Optional.empty();
            }
            return Optional.of(trainSession);
        }
    }

    /**
     * Get model feature maps.
     *
     * @return model weights.
     */
    public List<MSTensor> getFeatures() {
        if (trainSession == null) {
            return new ArrayList<>();
        }
        return trainSession.getFeaturesMap();
    }

    /**
     * update model feature maps.
     *
     * @param modelName model file name.
     * @param featureMaps new weights.
     * @return update status.
     */
    public Status updateFeatures(String modelName, List<FeatureMap> featureMaps) {
        if (trainSession == null || featureMaps == null || modelName == null || modelName.isEmpty()) {
            logger.severe(Common.addTag("trainSession,featureMaps modelName cannot be null"));
            return Status.NULLPTR;
        }
        List<MSTensor> tensors = new ArrayList<>(featureMaps.size());
        for (FeatureMap newFeature : featureMaps) {
            if (newFeature == null) {
                logger.severe(Common.addTag("newFeature cannot be null"));
                return Status.NULLPTR;
            }
            ByteBuffer by = newFeature.dataAsByteBuffer();
            ByteBuffer newData = ByteBuffer.allocateDirect(by.remaining());
            newData.order(ByteOrder.nativeOrder());
            newData.put(by);
            tensors.add(new MSTensor(newFeature.weightFullname(), newData));
        }
        boolean isSuccess = trainSession.updateFeatures(tensors);
        for (MSTensor tensor : tensors) {
            if (tensor == null) {
                logger.severe(Common.addTag("tensor cannot be null"));
                return Status.NULLPTR;
            }
            tensor.free();
        }

        if (isSuccess) {
            trainSession.export(modelName, 0, 0);
            return Status.SUCCESS;
        }
        return Status.FAILED;
    }

    /**
     * Free client.
     */
    public void free() {
        if (trainSession == null) {
            return;
        }
        trainSession.free();
        trainSession = null;
    }

    /**
     * Set learning rate.
     *
     * @param lr learning rate.
     * @return execute status.
     */
    public Status setLearningRate(float lr) {
        if (trainSession.setLearningRate(lr)) {
            return Status.SUCCESS;
        }
        logger.severe(Common.addTag("set learning rate failed"));
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
}