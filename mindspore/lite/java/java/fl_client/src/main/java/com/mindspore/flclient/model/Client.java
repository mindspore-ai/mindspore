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
import com.mindspore.config.MSContext;
import com.mindspore.config.ModelType;
import com.mindspore.config.TrainCfg;
import com.mindspore.flclient.FLParameter;
import com.mindspore.flclient.LocalFLParameter;
import com.mindspore.flclient.ServerMod;
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
     * Mindspore model proxy of train/infer
     */
    private ModelProxy trainModelProxy = null;
    private ModelProxy inferModelProxy = null;
    private ModelProxy curProxy = null;

    /**
     * dataset map.
     */
    public Map<RunType, DataSet> dataSets = new HashMap<>();

    private final List<ByteBuffer> inputsBuffer = new ArrayList<>();

    private float uploadLoss = 0.0f;

    /**
     * feature map before train
     */
    private Map<String, float[]> preFeatures = null;
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

    public Status initModel(FLParameter flParameter) {
        String trainModelPath = flParameter.getTrainModelPath();
        String inferModelPath = flParameter.getInferModelPath();
        int[][] inputShapes = flParameter.getInputShape();

        Status initRet = Status.FAILED;
        if (trainModelPath != null && !trainModelPath.isEmpty()) {
            trainModelProxy = new ModelProxy();
            initRet = trainModelProxy.initModel(trainModelPath, inputShapes);
        }

        if (inferModelPath == null || inferModelPath.isEmpty()) {
            return initRet;
        }

        // if infer/train modelpath is same, using the sameproxy to same memory
        if (inferModelPath.equals(trainModelPath)) {
            inferModelProxy = trainModelProxy;
        } else {
            inferModelProxy = new ModelProxy();
            initRet = inferModelProxy.initModel(inferModelPath, inputShapes);
        }
        return initRet;
    }

    public boolean EnableTrain(boolean enableFlg) {
        if (enableFlg == true && trainModelProxy != null) {
            curProxy = trainModelProxy;
            model = trainModelProxy.getModel();
            return true;
        }
        if (enableFlg == false && inferModelProxy != null) {
            curProxy = inferModelProxy;
            model = inferModelProxy.getModel();
            return true;
        }
        logger.severe("Can't get Model proxy for " + (enableFlg ? "train" : "infer"));
        return false;
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
        // Backup the prev Features for encrypt/compress
        preFeatures = curProxy.getFeatureMap();
        DataSet trainDataSet = dataSets.getOrDefault(RunType.TRAINMODE, null);
        if (trainDataSet == null) {
            logger.severe("not find train dataset");
            return Status.NULLPTR;
        }
        trainDataSet.padding();
        List<Callback> trainCallbacks = initCallbacks(RunType.TRAINMODE, trainDataSet);
        model.setTrainMode(true);
        Status status = curProxy.runModel(epochs, trainCallbacks, trainDataSet);
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
        Status status = curProxy.runModel(1, evalCallbacks, evalDataSet);
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
        Status status = curProxy.runModel(1, inferCallbacks, inferDataSet);
        if (status != Status.SUCCESS) {
            logger.severe("train loop failed");
            return null;
        }
        return getInferResult(inferCallbacks);
    }

    /**
     * Save model.
     * @param flParameter
     * @param localFLParameter
     * @return save result
     */
    public Status saveModel(FLParameter flParameter, LocalFLParameter localFLParameter) {
        String trainModelPath = flParameter.getTrainModelPath();
        String inferModelPath = flParameter.getInferModelPath();
        boolean exportRet = true;
        if (trainModelPath == null || trainModelPath.isEmpty() ||
                trainModelProxy == null || trainModelProxy.getModel() == null) {
            logger.info("No train model provided, no need save train model");
        } else {
            Model trainModel = trainModelProxy.getModel();
            exportRet = trainModel.export(trainModelPath, 0, false, null);
        }

        if (!exportRet) {
            logger.severe("Save train model to file failed.");
            return Status.FAILED;
        }

        if (!localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
            return Status.SUCCESS;
        }

        if (inferModelPath == null || inferModelPath.isEmpty() ||
                inferModelProxy == null || inferModelProxy.getModel() == null) {
            logger.info("No infer model provided, no need save infer model");
        } else {
            Model inferModel = inferModelProxy.getModel();
            exportRet = inferModel.export(inferModelPath, 0, false, null);
        }
        if (!exportRet) {
            logger.severe("Save infer model to file failed.");
            return Status.FAILED;
        }
        return Status.SUCCESS;
    }


    /**
     * calculate Weight normal for dp.
     *
     * @return model weights.
     */
    public float getDpWeightNorm(ArrayList<String> featureList) {
        if (preFeatures == null) {
            throw new RuntimeException("Must call getDpWeightNorm after train.");
        }
        float updateL2Norm = 0f;
        for (String key : featureList) {
            float[] preData = preFeatures.get(key);
            float[] curData = trainModelProxy.getFeature(key);
            if (preData == null || curData == null) {
                throw new RuntimeException("Get feature value failed, feature name:" + key);
            }

            if (preData.length != curData.length) {
                throw new RuntimeException("Length of " + key + " is changed after update, origin len:" +
                        preData.length + " cur len:" + curData.length);
            }
            for (int j = 0; j < preData.length; j++) {
                float updateData = preData[j] - curData[j];
                updateL2Norm += updateData * updateData;
            }
        }
        updateL2Norm = (float) Math.sqrt(updateL2Norm);
        return updateL2Norm;
    }

    /**
     * Get model feature with name.
     *
     * @return model weight.
     */
    public float[] getFeature(String weightName) {
        return curProxy.getFeature(weightName);
    }

    /**
     * Get model feature with name.
     *
     * @return model weight.
     */
    public float[] getPreFeature(String weightName) {
        return preFeatures.get(weightName);
    }

    /**
     * update model feature
     *
     * @param newFeature new weights.
     * @return update status.
     */
    public Status updateFeature(FeatureMap newFeature, boolean trainFlg) {
        if(trainFlg && trainModelProxy != null){
           return trainModelProxy.updateFeature(newFeature);
        }
        if(!trainFlg && inferModelProxy != null){
            return inferModelProxy.updateFeature(newFeature);
        }
        logger.severe("Can't get ModelProxy for " + (trainFlg?"trainModel":"inferModel"));
        return Status.FAILED;
    }

    /**
     * Free client.
     */
    public void free() {
        if (trainModelProxy != null) {
            trainModelProxy.free();
        }

        if (inferModelProxy != null && inferModelProxy != trainModelProxy) {
            inferModelProxy.free();
        }
        trainModelProxy = null;
        inferModelProxy = null;
        curProxy = null;
        model = null;
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
        return curProxy.getUploadLoss();
    }

    public void setUploadLoss(float uploadLoss) {
        curProxy.setUploadLoss(uploadLoss);
    }
}