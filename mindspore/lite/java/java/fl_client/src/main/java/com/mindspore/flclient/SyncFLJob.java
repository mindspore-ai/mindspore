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

import static com.mindspore.flclient.FLParameter.MAX_WAIT_TRY_TIME;
import static com.mindspore.flclient.FLParameter.RESTART_TIME_PER_ITER;
import static com.mindspore.flclient.FLParameter.SLEEP_TIME;
import static com.mindspore.flclient.LocalFLParameter.ALBERT;
import static com.mindspore.flclient.LocalFLParameter.ANDROID;
import static com.mindspore.flclient.LocalFLParameter.LENET;

import com.mindspore.flclient.common.FLLoggerGenerater;
import com.mindspore.flclient.model.AlInferBert;
import com.mindspore.flclient.model.AlTrainBert;
import com.mindspore.flclient.model.Client;
import com.mindspore.flclient.model.ClientManager;
import com.mindspore.flclient.model.RunType;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.Status;
import com.mindspore.flclient.model.TrainLenet;
import com.mindspore.flclient.pki.PkiUtil;
import com.mindspore.lite.config.CpuBindMode;
import mindspore.schema.ResponseGetModel;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.*;

/**
 * SyncFLJob defines the APIs for federated learning task.
 * API flJobRun() for starting federated learning on the device, the API modelInference() for inference on the
 * device, and the API getModel() for obtaining the latest model on the cloud.
 *
 * @since 2021-06-30
 */
public class SyncFLJob {
    private static Logger LOGGER = FLLoggerGenerater.getModelLogger(SyncFLJob.class.toString());
    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private IFLJobResultCallback flJobResultCallback;
    private FLClientStatus curStatus;
    private int tryTimePerIter = 0;
    private int lastIteration = -1;
    private int waitTryTime = 0;

    private void initFlIDForPkiVerify() {
        if (flParameter.isPkiVerify()) {
            LOGGER.info("pkiVerify mode is open!");
            String equipCertHash = PkiUtil.genEquipCertHash(flParameter.getClientID());
            if (equipCertHash == null || equipCertHash.isEmpty()) {
                LOGGER.severe("equipCertHash is empty, please check your mobile phone, only Huawei " +
                        "phones are supported now.");
                throw new IllegalArgumentException();
            }
            LOGGER.info("flID for pki verify is: " + equipCertHash);
            localFLParameter.setFlID(equipCertHash);
        } else {
            LOGGER.info("pkiVerify mode is not open!");
            localFLParameter.setFlID(flParameter.getClientID());
        }
    }

    public SyncFLJob() {
        if (!Common.checkFLName(flParameter.getFlName())) {
            try {
                LOGGER.info("the flName: " + flParameter.getFlName());
                Class.forName(flParameter.getFlName());
            } catch (ClassNotFoundException e) {
                LOGGER.severe("catch ClassNotFoundException error, the set flName does not exist, " +
                        "please " +
                        "check: " + e.getMessage());
                throw new IllegalArgumentException();
            }
        }
    }

    /**
     * Starts a federated learning task on the device.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus flJobRun() {
        flJobResultCallback = flParameter.getIflJobResultCallback();
        if (!Common.checkFLName(flParameter.getFlName()) && ANDROID.equals(flParameter.getDeployEnv())) {
            Common.setSecureRandom(Common.getFastSecureRandom());
        } else {
            Common.setSecureRandom(new SecureRandom());
        }
        initFlIDForPkiVerify();
        localFLParameter.setMsConfig(0, flParameter.getThreadNum(), flParameter.getCpuBindMode(), false);
        FLLiteClient flLiteClient = new FLLiteClient();
        LOGGER.info("recovery StopJobFlag to false in the start of fl job");
        localFLParameter.setStopJobFlag(false);
        InitialParameters();
        LOGGER.info("flJobRun start");
        flRunLoop(flLiteClient);
        LOGGER.info("flJobRun finish");
        flJobResultCallback.onFlJobFinished(flParameter.getFlName(), flLiteClient.getIterations(),
                flLiteClient.getRetCode());
        return curStatus;
    }

    private void flRunLoop(FLLiteClient flLiteClient) {
        do {
            if (tryTimeExceedsLimit() || checkStopJobFlag()) {
                break;
            }
            LOGGER.info("flName: " + flParameter.getFlName());
            int trainDataSize = flLiteClient.setInput();
            if (trainDataSize <= 0) {
                curStatus = FLClientStatus.FAILED;
                failed("unsolved error code in <flLiteClient.setInput>: the return trainDataSize<=0, setInput",
                        flLiteClient);
                break;
            }
            flLiteClient.setTrainDataSize(trainDataSize);

            // startFLJob
            curStatus = flLiteClient.startFLJob();
            if (curStatus == FLClientStatus.RESTART) {
                tryTimePerIter += 1;
                resetContext("[startFLJob]", flLiteClient.getNextRequestTime(), flLiteClient);
                continue;
            } else if (curStatus != FLClientStatus.SUCCESS) {
                failed("[startFLJob]", flLiteClient);
                break;
            }
            LOGGER.info("[startFLJob] startFLJob succeed, curIteration: " + flLiteClient.getIteration());
            updateTryTimePerIter(flLiteClient);

            // Copy weights before training.
            Map<String, float[]> oldFeatureMap = flLiteClient.getFeatureMap();
            localFLParameter.setOldFeatureMap(oldFeatureMap);

            // create mask
            curStatus = flLiteClient.getFeatureMask();
            if (curStatus == FLClientStatus.RESTART) {
                resetContext("[Encrypt] creatMask", flLiteClient.getNextRequestTime(), flLiteClient);
                continue;
            } else if (curStatus != FLClientStatus.SUCCESS) {
                failed("[Encrypt] createMask", flLiteClient);
                break;
            }

            // train
            curStatus = flLiteClient.localTrain();
            if (curStatus != FLClientStatus.SUCCESS) {
                failed("[train] train", flLiteClient);
                break;
            }
            LOGGER.info("[train] train succeed");

            // updateModel
            curStatus = flLiteClient.updateModel();
            if (curStatus == FLClientStatus.RESTART) {
                resetContext("[updateModel]", flLiteClient.getNextRequestTime(), flLiteClient);
                continue;
            } else if (curStatus != FLClientStatus.SUCCESS) {
                failed("[updateModel] updateModel", flLiteClient);
                break;
            }

            // unmasking
            curStatus = flLiteClient.unMasking();
            if (curStatus == FLClientStatus.RESTART) {
                resetContext("[Encrypt] unmasking", flLiteClient.getNextRequestTime(), flLiteClient);
                continue;
            } else if (curStatus != FLClientStatus.SUCCESS) {
                failed("[Encrypt] unmasking", flLiteClient);
                break;
            }

            // getModel
            curStatus = getModel(flLiteClient);
            if (curStatus == FLClientStatus.RESTART) {
                resetContext("[getModel]", flLiteClient.getNextRequestTime(), flLiteClient);
                continue;
            } else if (curStatus != FLClientStatus.SUCCESS) {
                failed("[getModel] getModel", flLiteClient);
                break;
            }

            // get the feature map after averaging and update dp_norm_clip
            flLiteClient.updateDpNormClip();

            // evaluate model after getting model from server
            if (!checkEvalPath()) {
                LOGGER.info("[evaluate] the data map set by user do not contain evaluation dataset, " +
                        "don't evaluate the model after getting model from server");
            } else {
                curStatus = flLiteClient.evaluateModel();
                if (curStatus != FLClientStatus.SUCCESS) {
                    failed("[evaluate] evaluate", flLiteClient);
                    break;
                }
                LOGGER.info("[evaluate] evaluate succeed");
            }
            LOGGER.info("========================================================the total response of "
                    + flLiteClient.getIteration() + ": " + curStatus +
                    "======================================================================");
            flJobResultCallback.onFlJobIterationFinished(flParameter.getFlName(), flLiteClient.getIteration(),
                    flLiteClient.getRetCode());
            Common.freeSession();
            tryTimePerIter = 0;
        } while (flLiteClient.getIteration() < flLiteClient.getIterations());
    }


    private void InitialParameters() {
        tryTimePerIter = 0;
        lastIteration = -1;
        waitTryTime = 0;
    }

    private Boolean tryTimeExceedsLimit() {
        if (tryTimePerIter > RESTART_TIME_PER_ITER) {
            LOGGER.severe("[tryTimeExceedsLimit] the repeated request time exceeds the limit, current" +
                    " repeated" +
                    " request time is: " + tryTimePerIter + " the limited time is: " + RESTART_TIME_PER_ITER);
            curStatus = FLClientStatus.FAILED;
            return true;
        }
        return false;
    }

    private void updateTryTimePerIter(FLLiteClient flLiteClient) {
        if (lastIteration != -1 && lastIteration == flLiteClient.getIteration()) {
            tryTimePerIter += 1;
        } else {
            tryTimePerIter = 1;
            lastIteration = flLiteClient.getIteration();
        }
    }

    private Boolean waitTryTimeExceedsLimit() {
        if (waitTryTime > MAX_WAIT_TRY_TIME) {
            LOGGER.severe("[waitTryTimeExceedsLimit] the waitTryTime exceeds the limit, current " +
                    "waitTryTime is: " + waitTryTime + " the limited time is: " + MAX_WAIT_TRY_TIME);
            curStatus = FLClientStatus.FAILED;
            return true;
        }
        return false;
    }

    private FLClientStatus getModel(FLLiteClient flLiteClient) {
        FLClientStatus curStatus = flLiteClient.getModel();
        waitTryTime = 0;
        while (curStatus == FLClientStatus.WAIT) {
            waitTryTime += 1;
            if (waitTryTimeExceedsLimit()) {
                curStatus = FLClientStatus.FAILED;
                break;
            }
            if (checkStopJobFlag()) {
                curStatus = FLClientStatus.FAILED;
                break;
            }
            waitSomeTime();
            curStatus = flLiteClient.getModel();
        }
        return curStatus;
    }

    private boolean checkEvalPath() {
        boolean tag = true;
        if (Common.checkFLName(flParameter.getFlName())) {
            if ("null".equals(flParameter.getTestDataset())) {
                tag = false;
            }
            return tag;
        }
        if (!flParameter.getDataMap().containsKey(RunType.EVALMODE)) {
            LOGGER.info("[evaluate] the data map set by user do not contain evaluation dataset, " +
                    "don't evaluate the model after getting model from server");
            tag = false;
            return tag;
        }
        return tag;
    }

    private boolean checkStopJobFlag() {
        if (localFLParameter.isStopJobFlag()) {
            LOGGER.info("the stopJObFlag is set to true, the job will be stop");
            curStatus = FLClientStatus.FAILED;
            return true;
        } else {
            return false;
        }
    }


    /**
     * Starts an inference task on the device.
     *
     * @return the status code corresponding to the response message.
     */
    public int[] modelInference() {
        if (Common.checkFLName(flParameter.getFlName())) {
            LOGGER.warning(Common.LOG_DEPRECATED);
            return deprecatedModelInference();
        }
        return new int[0];
    }

    /**
     * Starts an inference task on the device.
     *
     * @return the status code corresponding to the response message.
     */
    public List<Object> modelInfer() {
        Client client = ClientManager.getClient(flParameter.getFlName());
        localFLParameter.setMsConfig(0, flParameter.getThreadNum(), flParameter.getCpuBindMode(), false);
        localFLParameter.setStopJobFlag(false);
        if (!(null == flParameter.getInputShape())) {
            LOGGER.info("[model inference] the inference model has dynamic input.");
        }
        Map<RunType, Integer> dataSize = client.initDataSets(flParameter.getDataMap());
        if (dataSize.isEmpty()) {
            LOGGER.severe("[model inference] initDataSets failed, please check");
            client.free();
            return null;
        }
        Status tag = client.initSessionAndInputs(flParameter.getInferModelPath(), localFLParameter.getMsConfig(),
                flParameter.getInputShape());
        if (!Status.SUCCESS.equals(tag)) {
            LOGGER.severe("[model inference] unsolved error code in <initSessionAndInputs>: the return " +
                    " status is: " + tag);
            client.free();
            return null;
        }
        client.setBatchSize(flParameter.getBatchSize());
        LOGGER.info("===========model inference=============");
        List<Object> labels = client.inferModel();
        if (labels == null || labels.size() == 0) {
            LOGGER.severe("[model inference] the returned label from client.inferModel() is null, please " +
                    "check");
            client.free();
            return null;
        }
        LOGGER.fine("[model inference] the predicted outputs: " + Arrays.deepToString(labels.toArray()));
        client.free();
        LOGGER.info("[model inference] inference finish");
        return labels;
    }

    /**
     * Obtains the latest model on the cloud.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus getModel() {
        if (!Common.checkFLName(flParameter.getFlName()) && ANDROID.equals(flParameter.getDeployEnv())) {
            Common.setSecureRandom(Common.getFastSecureRandom());
        } else {
            Common.setSecureRandom(new SecureRandom());
        }
        if (Common.checkFLName(flParameter.getFlName())) {
            return deprecatedGetModel();
        }
        localFLParameter.setServerMod(flParameter.getServerMod().toString());
        localFLParameter.setMsConfig(0, 1, 0, false);
        FLClientStatus status;
        FLLiteClient flLiteClient = new FLLiteClient();
        status = flLiteClient.getModel();
        Common.freeSession();
        return status;
    }

    /**
     * use to stop FL job.
     */
    public void stopFLJob() {
        LOGGER.info("will stop the flJob");
        localFLParameter.setStopJobFlag(true);
        Common.notifyObject();
    }

    private void waitSomeTime() {
        if (flParameter.getSleepTime() != 0) {
            Common.sleep(flParameter.getSleepTime());
        } else {
            Common.sleep(SLEEP_TIME);
        }
    }

    private void waitNextReqTime(String nextReqTime) {
        long waitTime = Common.getWaitTime(nextReqTime);
        Common.sleep(waitTime);
    }

    private void resetContext(String tag, String nextReqTime, FLLiteClient flLiteClient) {
        LOGGER.info(tag + " out of time: need wait and request startFLJob again");
        waitNextReqTime(nextReqTime);
        Common.freeSession();
        flJobResultCallback.onFlJobIterationFinished(flParameter.getFlName(), flLiteClient.getIteration(),
                flLiteClient.getRetCode());
    }

    private void failed(String tag, FLLiteClient flLiteClient) {
        LOGGER.info(tag + " failed");
        LOGGER.info("=========================================the total response of " +
                flLiteClient.getIteration() + ": " + curStatus + "=========================================");
        Common.freeSession();
        flJobResultCallback.onFlJobIterationFinished(flParameter.getFlName(), flLiteClient.getIteration(),
                flLiteClient.getRetCode());
    }

    private static Map<RunType, List<String>> createDatasetMap(String trainDataPath, String evalDataPath,
                                                               String inferDataPath, String pathRegex) {
        Map<RunType, List<String>> dataMap = new HashMap<>();
        if ((trainDataPath == null) || ("null".equals(trainDataPath)) || (trainDataPath.isEmpty())) {
            LOGGER.info("the trainDataPath is null or empty, please check if you are in the case of " +
                    "only inference");
        } else {
            dataMap.put(RunType.TRAINMODE, Arrays.asList(trainDataPath.split(pathRegex)));
            LOGGER.info("the trainDataPath: " + Arrays.toString(trainDataPath.split(pathRegex)));
        }

        if ((evalDataPath == null) || ("null".equals(evalDataPath)) || (evalDataPath.isEmpty())) {
            LOGGER.info("the evalDataPath is null or empty, please check if you are in the case of only" +
                    " training without evaluation");
        } else {
            dataMap.put(RunType.EVALMODE, Arrays.asList(evalDataPath.split(pathRegex)));
            LOGGER.info("the evalDataPath: " + Arrays.toString(evalDataPath.split(pathRegex)));
        }

        if ((inferDataPath == null) || ("null".equals(inferDataPath)) || (inferDataPath.isEmpty())) {
            LOGGER.info("the inferDataPath is null or empty, please check if you are in the case of " +
                    "training without inference");
        } else {
            dataMap.put(RunType.INFERMODE, Arrays.asList(inferDataPath.split(pathRegex)));
            LOGGER.info("the inferDataPath: " + Arrays.toString(inferDataPath.split(pathRegex)));
        }
        return dataMap;
    }

    private static void createWeightNameList(String trainWeightName, String inferWeightName, String nameRegex,
                                             FLParameter flParameter) {
        if ((trainWeightName == null) || ("null".equals(trainWeightName)) || (trainWeightName.isEmpty())) {
            LOGGER.info("the trainWeightName is null or empty, only need in " + ServerMod.HYBRID_TRAINING);
        } else {
            flParameter.setHybridWeightName(Arrays.asList(trainWeightName.split(nameRegex)), RunType.TRAINMODE);
            LOGGER.info("the trainWeightName: " + Arrays.toString(trainWeightName.split(nameRegex)));
        }

        if ((inferWeightName == null) || ("null".equals(inferWeightName)) || (inferWeightName.isEmpty())) {
            LOGGER.info("the inferWeightName is null or empty, only need in " + ServerMod.HYBRID_TRAINING);
        } else {
            flParameter.setHybridWeightName(Arrays.asList(inferWeightName.split(nameRegex)), RunType.INFERMODE);
            LOGGER.info("the trainWeightName: " + Arrays.toString(inferWeightName.split(nameRegex)));
        }
    }

    private static int[][] getInputShapeArray(String inputShape) {
        String[] inputs = inputShape.split(";");
        int inputsSize = inputs.length;
        int[][] inputsArray = new int[inputsSize][];
        for (int i = 0; i < inputsSize; i++) {
            String[] input = inputs[i].split(",");
            int[] inputArray = Arrays.stream(input).mapToInt(Integer::parseInt).toArray();
            inputsArray[i] = inputArray;
        }
        return inputsArray;
    }

    private static void task(String[] args) {
        String trainDataPath = args[0];
        String evalDataPath = args[1];
        String inferDataPath = args[2];
        String pathRegex = args[3];

        String flName = args[4];
        String trainModelPath = args[5];
        String inferModelPath = args[6];
        String sslProtocol = args[7];
        String deployEnv = args[8];
        String domainName = args[9];
        String certPath = args[10];
        boolean useElb = Boolean.parseBoolean(args[11]);
        int serverNum = Integer.parseInt(args[12]);
        String task = args[13];
        int threadNum = Integer.parseInt(args[14]);
        String cpuBindMode = args[15];
        String trainWeightName = args[16];
        String inferWeightName = args[17];
        String nameRegex = args[18];
        String serverMod = args[19];
        String inputShape = args[21];
        int batchSize = Integer.parseInt(args[20]);
        FLParameter flParameter = FLParameter.getInstance();

        if (!("null".equals(inputShape) || inputShape == null)) {
            flParameter.setInputShape(getInputShapeArray(inputShape));
        }

        // create dataset of map
        Map<RunType, List<String>> dataMap = createDatasetMap(trainDataPath, evalDataPath, inferDataPath, pathRegex);

        // create weight name of list
        createWeightNameList(trainWeightName, inferWeightName, nameRegex, flParameter);

        flParameter.setFlName(flName);
        SyncFLJob syncFLJob = new SyncFLJob();
        switch (task) {
            case "train":
                LOGGER.info("start syncFLJob.flJobRun()");
                flParameter.setDataMap(dataMap);
                flParameter.setTrainModelPath(trainModelPath);
                flParameter.setInferModelPath(inferModelPath);
                flParameter.setSslProtocol(sslProtocol);
                flParameter.setDeployEnv(deployEnv);
                flParameter.setDomainName(domainName);
                if (Common.isHttps()) {
                    flParameter.setCertPath(certPath);
                }
                flParameter.setUseElb(useElb);
                flParameter.setServerNum(serverNum);
                flParameter.setThreadNum(threadNum);
                flParameter.setCpuBindMode(BindMode.valueOf(cpuBindMode));
                flParameter.setBatchSize(batchSize);
                syncFLJob.flJobRun();
                break;
            case "inference":
                LOGGER.info("start syncFLJob.modelInference()");
                flParameter.setDataMap(dataMap);
                flParameter.setInferModelPath(inferModelPath);
                flParameter.setThreadNum(threadNum);
                flParameter.setCpuBindMode(BindMode.valueOf(cpuBindMode));
                flParameter.setBatchSize(batchSize);
                syncFLJob.modelInfer();
                break;
            case "getModel":
                LOGGER.info("start syncFLJob.getModel()");
                flParameter.setTrainModelPath(trainModelPath);
                flParameter.setInferModelPath(inferModelPath);
                flParameter.setSslProtocol(sslProtocol);
                flParameter.setDeployEnv(deployEnv);
                flParameter.setDomainName(domainName);
                if (Common.isHttps()) {
                    flParameter.setCertPath(certPath);
                }
                flParameter.setUseElb(useElb);
                flParameter.setServerNum(serverNum);
                flParameter.setServerMod(ServerMod.valueOf(serverMod));
                syncFLJob.getModel();
                break;
            default:
                LOGGER.info("do not do any thing!");
        }
    }


    private static void deprecatedTask(String[] args) {
        String trainDataset = args[0];
        String vocabFile = args[1];
        String idsFile = args[2];
        String testDataset = args[3];
        String flName = args[4];
        String trainModelPath = args[5];
        String inferModelPath = args[6];
        boolean useSSL = Boolean.parseBoolean(args[7]);
        String domainName = args[8];
        boolean useElb = Boolean.parseBoolean(args[9]);
        int serverNum = Integer.parseInt(args[10]);
        String certPath = args[11];
        String task = args[12];

        FLParameter flParameter = FLParameter.getInstance();
        flParameter.setFlName(flName);
        SyncFLJob syncFLJob = new SyncFLJob();
        switch (task) {
            case "train":
                LOGGER.info("start syncFLJob.flJobRun()");
                flParameter.setTrainDataset(trainDataset);
                flParameter.setTrainModelPath(trainModelPath);
                flParameter.setTestDataset(testDataset);
                flParameter.setInferModelPath(inferModelPath);
                flParameter.setDomainName(domainName);
                if (Common.isHttps()) {
                    flParameter.setCertPath(certPath);
                }
                flParameter.setUseElb(useElb);
                flParameter.setServerNum(serverNum);
                if (ALBERT.equals(flName)) {
                    flParameter.setVocabFile(vocabFile);
                    flParameter.setIdsFile(idsFile);
                }
                syncFLJob.flJobRun();
                break;
            case "inference":
                LOGGER.info("start syncFLJob.modelInference()");
                flParameter.setTestDataset(testDataset);
                flParameter.setInferModelPath(inferModelPath);
                if (ALBERT.equals(flName)) {
                    flParameter.setVocabFile(vocabFile);
                    flParameter.setIdsFile(idsFile);
                }
                syncFLJob.modelInference();
                break;
            case "getModel":
                LOGGER.info("start syncFLJob.getModel()");
                flParameter.setTrainModelPath(trainModelPath);
                flParameter.setInferModelPath(inferModelPath);
                flParameter.setDomainName(domainName);
                if (Common.isHttps()) {
                    flParameter.setCertPath(certPath);
                }
                flParameter.setUseElb(useElb);
                flParameter.setServerNum(serverNum);
                syncFLJob.getModel();
                break;
            default:
                LOGGER.info("do not do any thing!");
        }
    }

    private int[] deprecatedModelInference() {
        int[] labels = new int[0];
        if (flParameter.getFlName().equals(ALBERT)) {
            AlInferBert alInferBert = AlInferBert.getInstance();
            LOGGER.info("===========model inference=============");
            labels = alInferBert.inferModel(flParameter.getInferModelPath(), flParameter.getTestDataset(),
                    flParameter.getVocabFile(), flParameter.getIdsFile());
            if (labels == null || labels.length == 0) {
                LOGGER.severe("[model inference] the returned label from adInferBert.inferModel() is null, please " +
                        "check");
            }
            LOGGER.info("[model inference] the predicted labels: " + Arrays.toString(labels));
            SessionUtil.free(alInferBert.getTrainSession());
            LOGGER.info("[model inference] inference finish");
        } else if (flParameter.getFlName().equals(LENET)) {
            TrainLenet trainLenet = TrainLenet.getInstance();
            LOGGER.info("===========model inference=============");
            labels = trainLenet.inferModel(flParameter.getInferModelPath(), flParameter.getTestDataset().split(",")[0]);
            if (labels == null || labels.length == 0) {
                LOGGER.severe("[model inference] the return labels is null.");
            }
            LOGGER.info("[model inference] the predicted labels: " + Arrays.toString(labels));
            SessionUtil.free(trainLenet.getTrainSession());
            LOGGER.info("[model inference] inference finish");
        }
        return labels;

    }

    private FLClientStatus deprecatedGetModel() {
        localFLParameter.setServerMod(ServerMod.FEDERATED_LEARNING.toString());
        FLClientStatus status;
        FLLiteClient flLiteClient = new FLLiteClient();
        status = flLiteClient.getModel();
        return status;
    }

    public static void main(String[] args) {
        if (args[4] == null || args[4].isEmpty()) {
            LOGGER.severe("the parameter of <args[4]> is null, please check");
            throw new IllegalArgumentException();
        }
        if (Common.checkFLName(args[4])) {
            deprecatedTask(args);
        } else {
            task(args);
        }

    }
}
