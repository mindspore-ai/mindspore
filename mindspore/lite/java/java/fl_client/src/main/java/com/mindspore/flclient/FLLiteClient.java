/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

package com.mindspore.flclient;

import static com.mindspore.flclient.FLParameter.SLEEP_TIME;
import static com.mindspore.flclient.LocalFLParameter.ALBERT;
import static com.mindspore.flclient.LocalFLParameter.LENET;

import com.mindspore.flclient.model.AlInferBert;
import com.mindspore.flclient.model.AlTrainBert;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.TrainLenet;

import mindspore.schema.CipherPublicParams;
import mindspore.schema.FLPlan;
import mindspore.schema.ResponseCode;
import mindspore.schema.ResponseFLJob;
import mindspore.schema.ResponseGetModel;
import mindspore.schema.ResponseUpdateModel;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Logger;

/**
 * Defining the general process of federated learning tasks.
 *
 * @since 2021-06-30
 */
public class FLLiteClient {
    private static final Logger LOGGER = Logger.getLogger(FLLiteClient.class.toString());
    private static int iteration = 0;
    private static Map<String, float[]> mapBeforeTrain;

    private double dpNormClipFactor = 1.0d;
    private double dpNormClipAdapt = 0.05d;
    private FLCommunication flCommunication;
    private FLClientStatus status;
    private int retCode;
    private int iterations = 1;
    private int epochs = 1;
    private int batchSize = 16;
    private int minSecretNum;
    private byte[] prime;
    private int featureSize;
    private int trainDataSize;
    private double dpEps = 100d;
    private double dpDelta = 0.01d;
    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private SecureProtocol secureProtocol = new SecureProtocol();
    private String nextRequestTime;

    /**
     * Defining a constructor of teh class FLLiteClient.
     */
    public FLLiteClient() {
        flCommunication = FLCommunication.getInstance();
    }

    private int setGlobalParameters(ResponseFLJob flJob) {
        FLPlan flPlan = flJob.flPlanConfig();
        if (flPlan == null) {
            LOGGER.severe(Common.addTag("[startFLJob] the FLPlan get from server is null"));
            return -1;
        }
        iterations = flPlan.iterations();
        epochs = flPlan.epochs();
        batchSize = flPlan.miniBatch();
        String serverMod = flPlan.serverMode();
        localFLParameter.setServerMod(serverMod);
        LOGGER.info(Common.addTag("[startFLJob] GlobalParameters <serverMod> from server: " + serverMod));
        if (localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
            LOGGER.info(Common.addTag("[startFLJob] set <batchSize> for AlTrainBert: " + batchSize));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            alTrainBert.setBatchSize(batchSize);
        } else if (localFLParameter.getServerMod().equals(ServerMod.FEDERATED_LEARNING.toString())) {
            LOGGER.info(Common.addTag("[startFLJob] set <batchSize> for TrainLenet: " + batchSize));
            TrainLenet trainLenet = TrainLenet.getInstance();
            trainLenet.setBatchSize(batchSize);
        } else {
            LOGGER.severe(Common.addTag("[startFLJob] the ServerMod returned from server is not valid"));
            return -1;
        }
        LOGGER.info(Common.addTag("[startFLJob] GlobalParameters <iterations> from server: " + iterations));
        LOGGER.info(Common.addTag("[startFLJob] GlobalParameters <epochs> from server: " + epochs));
        LOGGER.info(Common.addTag("[startFLJob] GlobalParameters <batchSize> from server: " + batchSize));
        CipherPublicParams cipherPublicParams = flPlan.cipher();
        if (cipherPublicParams == null) {
            LOGGER.severe(Common.addTag("[startFLJob] the cipherPublicParams returned from server is null"));
            return -1;
        }
        String encryptLevel = cipherPublicParams.encryptType();
        if (encryptLevel == null || encryptLevel.isEmpty()) {
            LOGGER.severe(Common.addTag("[startFLJob] GlobalParameters <encryptLevel> from server is null, set the " +
                    "encryptLevel to NOT_ENCRYPT "));
            localFLParameter.setEncryptLevel(EncryptLevel.NOT_ENCRYPT.toString());
        } else {
            localFLParameter.setEncryptLevel(encryptLevel);
            LOGGER.info(Common.addTag("[startFLJob] GlobalParameters <encryptLevel> from server: " + encryptLevel));
        }
        switch (localFLParameter.getEncryptLevel()) {
            case PW_ENCRYPT:
                minSecretNum = cipherPublicParams.t();
                int primeLength = cipherPublicParams.primeLength();
                prime = new byte[primeLength];
                for (int i = 0; i < primeLength; i++) {
                    prime[i] = (byte) cipherPublicParams.prime(i);
                }
                LOGGER.info(Common.addTag("[startFLJob] GlobalParameters <minSecretNum> from server: " + minSecretNum));
                if (minSecretNum <= 0) {
                    LOGGER.info(Common.addTag("[startFLJob] GlobalParameters <minSecretNum> from server is not valid:" +
                            "  <=0"));
                    return -1;
                }
                break;
            case DP_ENCRYPT:
                dpEps = cipherPublicParams.dpEps();
                dpDelta = cipherPublicParams.dpDelta();
                dpNormClipFactor = cipherPublicParams.dpNormClip();
                LOGGER.info(Common.addTag("[startFLJob] GlobalParameters <dpEps> from server: " + dpEps));
                LOGGER.info(Common.addTag("[startFLJob] GlobalParameters <dpDelta> from server: " + dpDelta));
                LOGGER.info(Common.addTag("[startFLJob] GlobalParameters <dpNormClipFactor> from server: " +
                        dpNormClipFactor));
                break;
            default:
                LOGGER.info(Common.addTag("[startFLJob] NOT_ENCRYPT, do not set parameter for Encrypt"));
        }
        return 0;
    }

    /**
     * Obtain retCode returned by server.
     *
     * @return the retCode returned by server.
     */
    public int getRetCode() {
        return retCode;
    }

    /**
     * Obtain current iteration returned by server.
     *
     * @return the current iteration returned by server.
     */
    public int getIteration() {
        return iteration;
    }

    /**
     * Obtain total iterations for the task returned by server.
     *
     * @return the total iterations for the task returned by server.
     */
    public int getIterations() {
        return iterations;
    }

    /**
     * Obtain the returned timestamp for next request from server.
     *
     * @return the timestamp for next request.
     */
    public String getNextRequestTime() {
        return nextRequestTime;
    }

    /**
     * set the size of train date set.
     *
     * @param trainDataSize the size of train date set.
     */
    public void setTrainDataSize(int trainDataSize) {
        this.trainDataSize = trainDataSize;
    }

    /**
     * Obtain the dpNormClipFactor.
     *
     * @return the dpNormClipFactor.
     */
    public double getDpNormClipFactor() {
        return dpNormClipFactor;
    }

    /**
     * Obtain the dpNormClipAdapt.
     *
     * @return the dpNormClipAdapt.
     */
    public double getDpNormClipAdapt() {
        return dpNormClipAdapt;
    }

    /**
     * Set the dpNormClipAdapt.
     *
     * @param dpNormClipAdapt the dpNormClipAdapt.
     */
    public void setDpNormClipAdapt(double dpNormClipAdapt) {
        this.dpNormClipAdapt = dpNormClipAdapt;
    }

    /**
     * Send serialized request message of startFLJob to server.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus startFLJob() {
        LOGGER.info(Common.addTag("[startFLJob] ====================================Verify " +
                "server===================================="));
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                flParameter.getDomainName());
        StartFLJob startFLJob = StartFLJob.getInstance();
        Date date = new Date();
        long time = date.getTime();
        byte[] msg = startFLJob.getRequestStartFLJob(trainDataSize, iteration, time);
        try {
            long start = Common.startTime("single startFLJob");
            LOGGER.info(Common.addTag("[startFLJob] the request message length: " + msg.length));
            byte[] message = flCommunication.syncRequest(url + "/startFLJob", msg);
            if (!Common.isSeverReady(message)) {
                LOGGER.info(Common.addTag("[startFLJob] the server is not ready now, need wait some time and request " +
                        "again"));
                status = FLClientStatus.RESTART;
                Common.sleep(SLEEP_TIME);
                nextRequestTime = "";
                return status;
            }
            LOGGER.info(Common.addTag("[startFLJob] the response message length: " + message.length));
            Common.endTime(start, "single startFLJob");
            ByteBuffer buffer = ByteBuffer.wrap(message);
            ResponseFLJob responseDataBuf = ResponseFLJob.getRootAsResponseFLJob(buffer);
            status = judgeStartFLJob(startFLJob, responseDataBuf);
            retCode = responseDataBuf.retcode();
        } catch (IOException e) {
            LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in StartFLJob: catch IOException: " +
                    e.getMessage()));
            status = FLClientStatus.FAILED;
            retCode = ResponseCode.RequestError;
        }
        return status;
    }

    private FLClientStatus judgeStartFLJob(StartFLJob startFLJob, ResponseFLJob responseDataBuf) {
        iteration = responseDataBuf.iteration();
        FLClientStatus response = startFLJob.doResponse(responseDataBuf);
        status = response;
        switch (response) {
            case SUCCESS:
                LOGGER.info(Common.addTag("[startFLJob] startFLJob success"));
                featureSize = startFLJob.getFeatureSize();
                secureProtocol.setEncryptFeatureName(startFLJob.getEncryptFeatureName());
                LOGGER.info(Common.addTag("[startFLJob] ***the feature size get in ResponseFLJob***: " + featureSize));
                int tag = setGlobalParameters(responseDataBuf);
                if (tag == -1) {
                    LOGGER.severe(Common.addTag("[startFLJob] setGlobalParameters failed"));
                    status = FLClientStatus.FAILED;
                }
                break;
            case RESTART:
                FLPlan flPlan = responseDataBuf.flPlanConfig();
                if (flPlan == null) {
                    LOGGER.severe(Common.addTag("[startFLJob] the flPlan returned from server is null"));
                    return FLClientStatus.FAILED;
                }
                iterations = flPlan.iterations();
                LOGGER.info(Common.addTag("[startFLJob] GlobalParameters <iterations> from server: " + iterations));
                nextRequestTime = responseDataBuf.nextReqTime();
                break;
            case FAILED:
                LOGGER.severe(Common.addTag("[startFLJob] startFLJob failed"));
                break;
            default:
                LOGGER.severe(Common.addTag("[startFLJob] failed: the response of startFLJob is out of range " +
                        "<SUCCESS, WAIT, FAILED, Restart>"));
                status = FLClientStatus.FAILED;
        }
        return status;
    }

    /**
     * Define the training process.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus localTrain() {
        LOGGER.info(Common.addTag("[train] ====================================global train epoch " + iteration +
                "===================================="));
        status = FLClientStatus.SUCCESS;
        retCode = ResponseCode.SUCCEED;
        if (flParameter.getFlName().equals(ALBERT)) {
            LOGGER.info(Common.addTag("[train] train in albert"));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            int tag = alTrainBert.trainModel(flParameter.getTrainModelPath(), epochs);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[train] unsolved error code in <alTrainBert.trainModel>"));
                status = FLClientStatus.FAILED;
                retCode = ResponseCode.RequestError;
            }
        } else if (flParameter.getFlName().equals(LENET)) {
            LOGGER.info(Common.addTag("[train] train in lenet"));
            TrainLenet trainLenet = TrainLenet.getInstance();
            int tag = trainLenet.trainModel(flParameter.getTrainModelPath(), epochs);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[train] unsolved error code in <trainLenet.trainModel>"));
                status = FLClientStatus.FAILED;
                retCode = ResponseCode.RequestError;
            }
        } else {
            LOGGER.severe(Common.addTag("[train] the flName is not valid"));
            status = FLClientStatus.FAILED;
            retCode = ResponseCode.RequestError;
        }
        return status;
    }

    /**
     * Send serialized request message of updateModel to server.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus updateModel() {
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                flParameter.getDomainName());
        UpdateModel updateModelBuf = UpdateModel.getInstance();
        byte[] updateModelBuffer = updateModelBuf.getRequestUpdateFLJob(iteration, secureProtocol, trainDataSize);
        if (updateModelBuf.getStatus() == FLClientStatus.FAILED) {
            LOGGER.info(Common.addTag("[updateModel] catch error in build RequestUpdateFLJob"));
            return FLClientStatus.FAILED;
        }
        try {
            long start = Common.startTime("single updateModel");
            LOGGER.info(Common.addTag("[updateModel] the request message length: " + updateModelBuffer.length));
            byte[] message = flCommunication.syncRequest(url + "/updateModel", updateModelBuffer);
            if (!Common.isSeverReady(message)) {
                LOGGER.info(Common.addTag("[updateModel] the server is not ready now, need wait some time and request" +
                        " again"));
                status = FLClientStatus.RESTART;
                Common.sleep(SLEEP_TIME);
                nextRequestTime = "";
                return status;
            }
            LOGGER.info(Common.addTag("[updateModel] the response message length: " + message.length));
            Common.endTime(start, "single updateModel");
            ByteBuffer debugBuffer = ByteBuffer.wrap(message);
            ResponseUpdateModel responseDataBuf = ResponseUpdateModel.getRootAsResponseUpdateModel(debugBuffer);
            status = updateModelBuf.doResponse(responseDataBuf);
            retCode = responseDataBuf.retcode();
            if (status == FLClientStatus.RESTART) {
                nextRequestTime = responseDataBuf.nextReqTime();
            }
            LOGGER.info(Common.addTag("[updateModel] get response from server ok!"));
        } catch (IOException e) {
            LOGGER.severe(Common.addTag("[updateModel] unsolved error code in updateModel: catch IOException: " +
                    e.getMessage()));
            status = FLClientStatus.FAILED;
            retCode = ResponseCode.RequestError;
        }
        return status;
    }

    /**
     * Send serialized request message of getModel to server.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus getModel() {
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                flParameter.getDomainName());
        GetModel getModelBuf = GetModel.getInstance();
        byte[] buffer = getModelBuf.getRequestGetModel(flParameter.getFlName(), iteration);
        try {
            long start = Common.startTime("single getModel");
            LOGGER.info(Common.addTag("[getModel] the request message length: " + buffer.length));
            byte[] message = flCommunication.syncRequest(url + "/getModel", buffer);
            if (!Common.isSeverReady(message)) {
                LOGGER.info(Common.addTag("[getModel] the server is not ready now, need wait some time and request " +
                        "again"));
                status = FLClientStatus.WAIT;
                return status;
            }
            LOGGER.info(Common.addTag("[getModel] the response message length: " + message.length));
            Common.endTime(start, "single getModel");
            LOGGER.info(Common.addTag("[getModel] get model request success"));
            ByteBuffer debugBuffer = ByteBuffer.wrap(message);
            ResponseGetModel responseDataBuf = ResponseGetModel.getRootAsResponseGetModel(debugBuffer);
            status = getModelBuf.doResponse(responseDataBuf);
            retCode = responseDataBuf.retcode();
            if (status == FLClientStatus.RESTART) {
                nextRequestTime = responseDataBuf.timestamp();
            }
            LOGGER.info(Common.addTag("[getModel] get response from server ok!"));
        } catch (IOException e) {
            LOGGER.severe(Common.addTag("[getModel] un sloved error code: catch IOException: " + e.getMessage()));
            status = FLClientStatus.FAILED;
            retCode = ResponseCode.RequestError;
        }
        return status;
    }

    /**
     * Obtain the weight of the model before training.
     *
     * @param map a map to store the weight of the model.
     * @return the weight.
     */
    public static synchronized Map<String, float[]> getOldMapCopy(Map<String, float[]> map) {
        if (mapBeforeTrain == null) {
            Map<String, float[]> copyMap = new TreeMap<>();
            for (String key : map.keySet()) {
                float[] data = map.get(key);
                int dataLen = data.length;
                float[] weights = new float[dataLen];
                if ((key.indexOf("Default") < 0) && (key.indexOf("nhwc") < 0) && (key.indexOf("moment") < 0) &&
                        (key.indexOf("learning") < 0)) {
                    for (int j = 0; j < dataLen; j++) {
                        float weight = data[j];
                        weights[j] = weight;
                    }
                    copyMap.put(key, weights);
                }
            }
            mapBeforeTrain = copyMap;
        } else {
            for (String key : map.keySet()) {
                float[] data = map.get(key);
                float[] copyData = mapBeforeTrain.get(key);
                int dataLen = data.length;
                if ((key.indexOf("Default") < 0) && (key.indexOf("nhwc") < 0) && (key.indexOf("moment") < 0) &&
                        (key.indexOf("learning") < 0)) {
                    for (int j = 0; j < dataLen; j++) {
                        copyData[j] = data[j];
                    }
                }
            }
        }
        return mapBeforeTrain;
    }

    /**
     * Obtain pairwise mask and individual mask.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus getFeatureMask() {
        FLClientStatus curStatus;
        switch (localFLParameter.getEncryptLevel()) {
            case PW_ENCRYPT:
                LOGGER.info(Common.addTag("[Encrypt] creating feature mask of <" +
                        localFLParameter.getEncryptLevel().toString() + ">"));
                secureProtocol.setPWParameter(iteration, minSecretNum, prime, featureSize);
                curStatus = secureProtocol.pwCreateMask();
                if (curStatus == FLClientStatus.RESTART) {
                    nextRequestTime = secureProtocol.getNextRequestTime();
                }
                retCode = secureProtocol.getRetCode();
                LOGGER.info(Common.addTag("[Encrypt] the response of create mask for <" +
                        localFLParameter.getEncryptLevel().toString() + "> : " + curStatus));
                return curStatus;
            case DP_ENCRYPT:
                Map<String, float[]> map = new HashMap<String, float[]>();
                if (flParameter.getFlName().equals(ALBERT)) {
                    AlTrainBert alTrainBert = AlTrainBert.getInstance();
                    map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(alTrainBert.getTrainSession()));
                } else if (flParameter.getFlName().equals(LENET)) {
                    TrainLenet trainLenet = TrainLenet.getInstance();
                    map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(trainLenet.getTrainSession()));
                }
                Map<String, float[]> copyMap = getOldMapCopy(map);
                curStatus = secureProtocol.setDPParameter(iteration, dpEps, dpDelta, dpNormClipAdapt, copyMap);
                retCode = ResponseCode.SUCCEED;
                if (curStatus != FLClientStatus.SUCCESS) {
                    LOGGER.info(Common.addTag("---Differential privacy init failed---"));
                    retCode = ResponseCode.RequestError;
                    return FLClientStatus.FAILED;
                }
                LOGGER.info(Common.addTag("[Encrypt] set parameters for DP_ENCRYPT!"));
                return FLClientStatus.SUCCESS;
            case NOT_ENCRYPT:
                retCode = ResponseCode.SUCCEED;
                LOGGER.info(Common.addTag("[Encrypt] don't mask model"));
                return FLClientStatus.SUCCESS;
            default:
                retCode = ResponseCode.SUCCEED;
                LOGGER.severe(Common.addTag("[Encrypt] The encrypt level is error, not encrypt by default"));
                return FLClientStatus.SUCCESS;
        }
    }

    /**
     * Reconstruct the secrets used for unmasking model weights.
     *
     * @return current status code in client.
     */
    public FLClientStatus unMasking() {
        FLClientStatus curStatus;
        switch (localFLParameter.getEncryptLevel()) {
            case PW_ENCRYPT:
                curStatus = secureProtocol.pwUnmasking();
                retCode = secureProtocol.getRetCode();
                LOGGER.info(Common.addTag("[Encrypt] the response of unmasking : " + curStatus));
                if (curStatus == FLClientStatus.RESTART) {
                    nextRequestTime = secureProtocol.getNextRequestTime();
                }
                return curStatus;
            case DP_ENCRYPT:
                LOGGER.info(Common.addTag("[Encrypt] DP_ENCRYPT do not need unmasking"));
                retCode = ResponseCode.SUCCEED;
                return FLClientStatus.SUCCESS;
            case NOT_ENCRYPT:
                LOGGER.info(Common.addTag("[Encrypt] haven't mask model"));
                retCode = ResponseCode.SUCCEED;
                return FLClientStatus.SUCCESS;
            default:
                LOGGER.severe(Common.addTag("[Encrypt] The encrypt level is error, not encrypt by default"));
                retCode = ResponseCode.SUCCEED;
                return FLClientStatus.SUCCESS;
        }
    }

    /**
     * Evaluate model after getting model from server.
     *
     * @return the status code in client.
     */
    public FLClientStatus evaluateModel() {
        status = FLClientStatus.SUCCESS;
        retCode = ResponseCode.SUCCEED;
        LOGGER.info(Common.addTag("===================================evaluate model after getting model from " +
                "server==================================="));
        if (flParameter.getFlName().equals(ALBERT)) {
            float acc = 0;
            if (localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
                LOGGER.info(Common.addTag("[evaluate] evaluateModel by " + localFLParameter.getServerMod()));
                AlInferBert alInferBert = AlInferBert.getInstance();
                int dataSize = alInferBert.initDataSet(flParameter.getTestDataset(), flParameter.getVocabFile(),
                        flParameter.getIdsFile(), true);
                if (dataSize <= 0) {
                    LOGGER.severe(Common.addTag("[evaluate] unsolved error code in <alInferBert.initDataSet>: the " +
                            "return dataSize<=0"));
                    status = FLClientStatus.FAILED;
                    retCode = ResponseCode.RequestError;
                    return status;
                }
                acc = alInferBert.evalModel();
            } else {
                LOGGER.info(Common.addTag("[evaluate] evaluateModel by " + localFLParameter.getServerMod()));
                AlTrainBert alTrainBert = AlTrainBert.getInstance();
                int dataSize = alTrainBert.initDataSet(flParameter.getTestDataset(), flParameter.getVocabFile(),
                        flParameter.getIdsFile());
                if (dataSize <= 0) {
                    LOGGER.severe(Common.addTag("[evaluate] unsolved error code in <alTrainBert.initDataSet>: the " +
                            "return dataSize<=0"));
                    status = FLClientStatus.FAILED;
                    retCode = ResponseCode.RequestError;
                    return status;
                }
                acc = alTrainBert.evalModel();
            }
            if (Float.isNaN(acc)) {
                LOGGER.severe(Common.addTag("[evaluate] unsolved error code in <evalModel>: the return acc is NAN"));
                status = FLClientStatus.FAILED;
                retCode = ResponseCode.RequestError;
                return status;
            }
            LOGGER.info(Common.addTag("[evaluate] modelPath: " + flParameter.getInferModelPath() + " dataPath: " +
                    flParameter.getTestDataset() + " vocabFile: " + flParameter.getVocabFile() +
                    " idsFile: " + flParameter.getIdsFile()));
            LOGGER.info(Common.addTag("[evaluate] evaluate acc: " + acc));
        } else if (flParameter.getFlName().equals(LENET)) {
            TrainLenet trainLenet = TrainLenet.getInstance();
            if (flParameter.getTestDataset().split(",").length < 2) {
                LOGGER.severe(Common.addTag("[evaluate] the set testDataPath for lenet is not valid, should be the " +
                        "format of <data.bin,label.bin> "));
                status = FLClientStatus.FAILED;
                retCode = ResponseCode.RequestError;
                return status;
            }
            int dataSize = trainLenet.initDataSet(flParameter.getTestDataset().split(",")[0],
                    flParameter.getTestDataset().split(",")[1]);
            if (dataSize <= 0) {
                LOGGER.severe(Common.addTag("[evaluate] unsolved error code in <trainLenet.initDataSet>: the return " +
                        "dataSize<=0"));
                status = FLClientStatus.FAILED;
                retCode = ResponseCode.RequestError;
                return status;
            }
            float acc = trainLenet.evalModel();
            if (Float.isNaN(acc)) {
                LOGGER.severe(Common.addTag("[evaluate] unsolved error code in <trainLenet.evalModel>: the return acc" +
                        " is NAN"));
                status = FLClientStatus.FAILED;
                retCode = ResponseCode.RequestError;
                return status;
            }
            LOGGER.info(Common.addTag("[evaluate] modelPath: " + flParameter.getInferModelPath() + " dataPath: " +
                    flParameter.getTestDataset().split(",")[0] + " labelPath: " +
                    flParameter.getTestDataset().split(",")[1]));
            LOGGER.info(Common.addTag("[evaluate] evaluate acc: " + acc));
        }
        return status;
    }

    /**
     * Set date path.
     *
     * @param dataPath, train or test dataset and label set.
     * @return date size.
     */
    public int setInput(String dataPath) {
        retCode = ResponseCode.SUCCEED;
        LOGGER.info(Common.addTag("==========set input==========="));
        int dataSize = 0;
        if (flParameter.getFlName().equals(ALBERT)) {
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            dataSize = alTrainBert.initDataSet(dataPath, flParameter.getVocabFile(), flParameter.getIdsFile());
            LOGGER.info(Common.addTag("[set input] " + "dataPath: " + dataPath + " dataSize: " + +dataSize + " " +
                    "vocabFile: " + flParameter.getVocabFile() + " idsFile: " + flParameter.getIdsFile()));
        } else if (flParameter.getFlName().equals(LENET)) {
            TrainLenet trainLenet = TrainLenet.getInstance();
            if (dataPath.split(",").length < 2) {
                LOGGER.severe(Common.addTag("[set input] the set dataPath for lenet is not valid, should be the " +
                        "format of <data.bin,label.bin> "));
                return -1;
            }
            dataSize = trainLenet.initDataSet(dataPath.split(",")[0], dataPath.split(",")[1]);
            LOGGER.info(Common.addTag("[set input] " + "dataPath: " + dataPath.split(",")[0] + " dataSize: " +
                    dataSize + " labelPath: " + dataPath.split(",")[1]));
        }
        if (dataSize <= 0) {
            retCode = ResponseCode.RequestError;
            return -1;
        }
        return dataSize;
    }

    /**
     * Initialization session.
     *
     * @return the status code in client.
     */
    public FLClientStatus initSession() {
        int tag = 0;
        retCode = ResponseCode.SUCCEED;
        if (flParameter.getFlName().equals(ALBERT)) {
            LOGGER.info(Common.addTag("==========Loading train model, " + flParameter.getTrainModelPath() + " Create " +
                    "Train Session============="));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            tag = alTrainBert.initSessionAndInputs(flParameter.getTrainModelPath(), true);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[initSession] unsolved error code in <initSessionAndInputs>: the return " +
                        "is -1"));
                retCode = ResponseCode.RequestError;
                return FLClientStatus.FAILED;
            }
            LOGGER.info(Common.addTag("==========Loading inference model, " + flParameter.getInferModelPath() + " " +
                    "Create inference Session============="));
            AlInferBert alInferBert = AlInferBert.getInstance();
            tag = alInferBert.initSessionAndInputs(flParameter.getInferModelPath(), false);
        } else if (flParameter.getFlName().equals(LENET)) {
            LOGGER.info(Common.addTag("==========Loading train model, " + flParameter.getTrainModelPath() + " Create " +
                    "Train Session============="));
            TrainLenet trainLenet = TrainLenet.getInstance();
            tag = trainLenet.initSessionAndInputs(flParameter.getTrainModelPath(), true);
        }
        if (tag == -1) {
            LOGGER.severe(Common.addTag("[initSession] unsolved error code in <initSessionAndInputs>: the return is " +
                    "-1"));
            retCode = ResponseCode.RequestError;
            return FLClientStatus.FAILED;
        }
        return FLClientStatus.SUCCESS;
    }

    /**
     * Free session.
     */
    protected void freeSession() {
        if (flParameter.getFlName().equals(ALBERT)) {
            LOGGER.info(Common.addTag("===========free train session============="));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            SessionUtil.free(alTrainBert.getTrainSession());
            if (!flParameter.getTestDataset().equals("null")) {
                LOGGER.info(Common.addTag("===========free inference session============="));
                AlInferBert alInferBert = AlInferBert.getInstance();
                SessionUtil.free(alInferBert.getTrainSession());
            }
        } else if (flParameter.getFlName().equals(LENET)) {
            LOGGER.info(Common.addTag("===========free session============="));
            TrainLenet trainLenet = TrainLenet.getInstance();
            SessionUtil.free(trainLenet.getTrainSession());
        }
    }
}
