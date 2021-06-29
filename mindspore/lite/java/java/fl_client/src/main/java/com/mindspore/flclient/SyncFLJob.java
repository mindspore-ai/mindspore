/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

import com.mindspore.flclient.model.AdInferBert;
import com.mindspore.flclient.model.AdTrainBert;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.TrainLenet;
import mindspore.schema.ResponseGetModel;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.logging.Logger;
import java.util.Map;
import java.util.HashMap;

import static com.mindspore.flclient.FLParameter.SLEEP_TIME;
import static com.mindspore.flclient.LocalFLParameter.ADBERT;
import static com.mindspore.flclient.LocalFLParameter.LENET;

public class SyncFLJob {
    private static final Logger LOGGER = Logger.getLogger(SyncFLJob.class.toString());
    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private FLJobResultCallback flJobResultCallback = new FLJobResultCallback();
    private Map<String, float[]> oldFeatureMap;

    public SyncFLJob() {
    }

    public void flJobRun() {
        localFLParameter.setFlID(flParameter.getClientID());
        FLLiteClient client = new FLLiteClient();
        client.initSession();

        FLClientStatus curStatus;
        do {
            LOGGER.info(Common.addTag("flName: " + flParameter.getFlName()));
            int trainDataSize = client.setInput(flParameter.getTrainDataset());
            LOGGER.info(Common.addTag("train path: " + flParameter.getTrainDataset()));
            LOGGER.info(Common.addTag("train data size: " + trainDataSize));
            if (trainDataSize <= 0) {
                LOGGER.severe(Common.addTag("unsolved error code in <client.setInput>: the return trainDataSize<=0"));
                curStatus = FLClientStatus.FAILED;
                flJobResultCallback.onFlJobIterationFinished(flParameter.getFlName(), client.getIteration(), client.getRetCode());
                break;
            }
            client.setTrainDataSize(trainDataSize);

            // startFLJob
            curStatus = client.startFLJob();
            while (curStatus == FLClientStatus.WAIT) {
                waitSomeTime();
                curStatus = client.startFLJob();
            }
            if (curStatus == FLClientStatus.RESTART) {
                restart("[startFLJob]", client.getNextRequestTime(), client.getIteration(), client.getRetCode());
                continue;
            } else if (curStatus == FLClientStatus.FAILED) {
                failed("[startFLJob]", client.getIteration(), client.getRetCode(), curStatus);
                break;
            }
            LOGGER.info(Common.addTag("[startFLJob] startFLJob succeed, curIteration: " + client.getIteration()));

            // get the feature map before train
            getOldFeatureMap(client);

            // create mask
            curStatus = client.getFeatureMask();
            if (curStatus == FLClientStatus.RESTART) {
                restart("[Encrypt] creatMask", client.getNextRequestTime(), client.getIteration(), client.getRetCode());
                continue;
            } else if (curStatus == FLClientStatus.FAILED) {
                failed("[Encrypt] createMask", client.getIteration(), client.getRetCode(), curStatus);
                break;
            }

            // train
            curStatus = client.localTrain();
            if (curStatus == FLClientStatus.FAILED) {
                failed("[train] train", client.getIteration(), client.getRetCode(), curStatus);
                break;
            }
            LOGGER.info(Common.addTag("[train] train succeed"));

            // updateModel
            curStatus = client.updateModel();
            while (curStatus == FLClientStatus.WAIT) {
                waitSomeTime();
                curStatus = client.updateModel();
            }
            if (curStatus == FLClientStatus.RESTART) {
                restart("[updateModel]", client.getNextRequestTime(), client.getIteration(), client.getRetCode());
                continue;
            } else if (curStatus == FLClientStatus.FAILED) {
                failed("[updateModel] updateModel", client.getIteration(), client.getRetCode(), curStatus);
                break;
            }
            LOGGER.info(Common.addTag("[updateModel] updateModel succeed"));

            // unmasking
            curStatus = client.unMasking();
            if (curStatus == FLClientStatus.RESTART) {
                restart("[Encrypt] unmasking", client.getNextRequestTime(), client.getIteration(), client.getRetCode());
                continue;
            } else if (curStatus == FLClientStatus.FAILED) {
                failed("[Encrypt] unmasking", client.getIteration(), client.getRetCode(), curStatus);
                break;
            }

            // getModel
            curStatus = client.getModel();
            while (curStatus == FLClientStatus.WAIT) {
                waitSomeTime();
                curStatus = client.getModel();
            }
            if (curStatus == FLClientStatus.RESTART) {
                restart("[getModel]", client.getNextRequestTime(), client.getIteration(), client.getRetCode());
                continue;
            } else if (curStatus == FLClientStatus.FAILED) {
                failed("[getModel] getModel", client.getIteration(), client.getRetCode(), curStatus);
                break;
            }
            LOGGER.info(Common.addTag("[getModel] getModel succeed"));

            // get the feature map after averaging and update dp_norm_clip
            updateDpNormClip(client);

            // evaluate model after getting model from server
            if (flParameter.getTestDataset().equals("null")) {
                LOGGER.info(Common.addTag("[evaluate] the testDataset is null, don't evaluate the combine model"));
            } else {
                curStatus = client.evaluateModel();
                if (curStatus == FLClientStatus.FAILED) {
                    failed("[evaluate] evaluate", client.getIteration(), client.getRetCode(), curStatus);
                    break;
                }
                LOGGER.info(Common.addTag("[evaluate] evaluate succeed"));
            }
            LOGGER.info(Common.addTag("========================================================the total response of " + client.getIteration() + ": " + curStatus + "======================================================================"));
            flJobResultCallback.onFlJobIterationFinished(flParameter.getFlName(), client.getIteration(), client.getRetCode());
        } while (client.getIteration() < client.getIterations());
        client.finalize();
        LOGGER.info(Common.addTag("flJobRun finish"));
        flJobResultCallback.onFlJobFinished(flParameter.getFlName(), client.getIterations(), client.getRetCode());
    }

    private void updateDpNormClip(FLLiteClient client) {
        EncryptLevel encryptLevel = localFLParameter.getEncryptLevel();
        if (encryptLevel == EncryptLevel.DP_ENCRYPT) {
            Map<String, float[]> fedFeatureMap = getFeatureMap();
            float fedWeightUpdateNorm = calWeightUpdateNorm(oldFeatureMap, fedFeatureMap);
            LOGGER.info(Common.addTag("[DP] L2-norm of weights' average update is: " + fedWeightUpdateNorm));
            client.dpNormClipAdapt = client.dpNormClipFactor*fedWeightUpdateNorm;
            LOGGER.info(Common.addTag("[DP] Adaptive dpNormClip is: " + client.dpNormClipAdapt));
        }
    }
    private void getOldFeatureMap(FLLiteClient client) {
        EncryptLevel encryptLevel = localFLParameter.getEncryptLevel();
        if (encryptLevel == EncryptLevel.DP_ENCRYPT) {
            Map<String, float[]> featureMap = getFeatureMap();
            oldFeatureMap = client.getOldMapCopy(featureMap);
        }
    }
    private float calWeightUpdateNorm(Map<String, float[]> originalData, Map<String, float[]> newData){
        float updateL2Norm = 0;
        for (String key: originalData.keySet()) {
            float[] data=originalData.get(key);
            float[] dataAfterUpdate = newData.get(key);
            for (int j = 0; j<data.length; j++) {
                float updateData = data[j] - dataAfterUpdate[j];
                updateL2Norm += updateData*updateData;
            }
        }
        updateL2Norm = (float)Math.sqrt(updateL2Norm);
        return updateL2Norm;
    }

    private Map<String, float[]> getFeatureMap() {
        Map<String, float[]> featureMap = new HashMap<>();
        if (flParameter.getFlName().equals(ADBERT)) {
            AdTrainBert adTrainBert = AdTrainBert.getInstance();
            featureMap = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(adTrainBert.getTrainSession()));
        } else if (flParameter.getFlName().equals(LENET)) {
            TrainLenet trainLenet = TrainLenet.getInstance();
            featureMap = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(trainLenet.getTrainSession()));
        }
        return featureMap;
    }

    public int[] modelInference(String flName, String dataPath, String vocabFile, String idsFile, String modelPath) {
        int[] labels = new int[0];
        if (flName.equals(ADBERT)) {
            AdInferBert adInferBert = AdInferBert.getInstance();
            LOGGER.info(Common.addTag("===========model inference============="));
            labels = adInferBert.inferModel(modelPath, dataPath, vocabFile, idsFile);
            LOGGER.info(Common.addTag("[model inference] the predicted labels: " + Arrays.toString(labels)));
            SessionUtil.free(adInferBert.getTrainSession());
            LOGGER.info(Common.addTag("[model inference] inference finish"));
        } else if (flName.equals(LENET)) {
            TrainLenet trainLenet = TrainLenet.getInstance();
            LOGGER.info(Common.addTag("===========model inference============="));
            labels = trainLenet.inferModel(modelPath, dataPath.split(",")[0]);
            LOGGER.info(Common.addTag("[model inference] the predicted labels: " + Arrays.toString(labels)));
            SessionUtil.free(trainLenet.getTrainSession());
            LOGGER.info(Common.addTag("[model inference] inference finish"));
        }
        if (labels.length == 0) {
            LOGGER.severe(Common.addTag("[model inference] the return labels is null."));
        }
        return labels;
    }

    public FLClientStatus getModel(boolean useElb, int serverNum, String ip, int port, String flName, String trainModelPath, String inferModelPath, boolean useSSL) {
        int tag = 0;
        flParameter.setTrainModelPath(trainModelPath);
        flParameter.setInferModelPath(inferModelPath);
        FLClientStatus status = FLClientStatus.SUCCESS;
        try {
            if (flName.equals(ADBERT)) {
                localFLParameter.setServerMod(ServerMod.HYBRID_TRAINING.toString());
                LOGGER.info(Common.addTag("[getModel] ==========Loading train model, " + trainModelPath + " Create Train Session============="));
                AdTrainBert adTrainBert = AdTrainBert.getInstance();
                tag = adTrainBert.initSessionAndInputs(trainModelPath, true);
                if (tag == -1) {
                    LOGGER.severe(Common.addTag("[initSession] unsolved error code in <initSessionAndInputs>: the return is -1"));
                    return FLClientStatus.FAILED;
                }
                LOGGER.info(Common.addTag("[getModel] ==========Loading inference model, " + inferModelPath + " Create inference Session============="));
                AdInferBert adInferBert = AdInferBert.getInstance();
                tag = adInferBert.initSessionAndInputs(inferModelPath, false);
            } else if (flName.equals(LENET)) {
                localFLParameter.setServerMod(ServerMod.FEDERATED_LEARNING.toString());
                LOGGER.info(Common.addTag("[getModel] ==========Loading train model, " + trainModelPath + " Create Train Session============="));
                TrainLenet trainLenet = TrainLenet.getInstance();
                tag = trainLenet.initSessionAndInputs(trainModelPath, true);
            }
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[initSession] unsolved error code in <initSessionAndInputs>: the return is -1"));
                return FLClientStatus.FAILED;
            }
            flParameter.setUseSSL(useSSL);
            FLCommunication flCommunication = FLCommunication.getInstance();
            String url = Common.generateUrl(flParameter.isUseHttps(), useElb, ip, port, serverNum);
            LOGGER.info(Common.addTag("[getModel] ===========getModel url: " + url + "=============="));
            GetModel getModelBuf = GetModel.getInstance();
            byte[] buffer = getModelBuf.getRequestGetModel(flName, 0);
            byte[] message = flCommunication.syncRequest(url + "/getModel", buffer);
            LOGGER.info(Common.addTag("[getModel] get model request success"));
            ByteBuffer debugBuffer = ByteBuffer.wrap(message);
            ResponseGetModel responseDataBuf = ResponseGetModel.getRootAsResponseGetModel(debugBuffer);
            status = getModelBuf.doResponse(responseDataBuf);
            LOGGER.info(Common.addTag("[getModel] success!"));
        } catch (Exception e) {
            LOGGER.severe(Common.addTag("[getModel] unsolved error code: catch Exception: " + e.getMessage()));
            status = FLClientStatus.FAILED;
        }
        if (flName.equals(ADBERT)) {
            LOGGER.info(Common.addTag("===========free train session============="));
            AdTrainBert adTrainBert = AdTrainBert.getInstance();
            SessionUtil.free(adTrainBert.getTrainSession());
            LOGGER.info(Common.addTag("===========free inference session============="));
            AdInferBert adInferBert = AdInferBert.getInstance();
            SessionUtil.free(adInferBert.getTrainSession());
        } else if (flName.equals(LENET)) {
            LOGGER.info(Common.addTag("===========free session============="));
            TrainLenet trainLenet = TrainLenet.getInstance();
            SessionUtil.free(trainLenet.getTrainSession());
        }
        return status;
    }

    private void waitSomeTime() {
        if (flParameter.getSleepTime() != 0)
            Common.sleep(flParameter.getSleepTime());
        else
            Common.sleep(SLEEP_TIME);
    }

    private void waitNextReqTime(String nextReqTime) {
        if (flParameter.isTimer()) {
            long waitTime = Common.getWaitTime(nextReqTime);
            Common.sleep(waitTime);
        } else {
            waitSomeTime();
        }
    }

    private void restart(String tag, String nextReqTime, int iteration, int retcode) {
        LOGGER.info(Common.addTag(tag + " out of time: need wait and request startFLJob again"));
        waitNextReqTime(nextReqTime);
        flJobResultCallback.onFlJobIterationFinished(flParameter.getFlName(), iteration, retcode);
    }

    private void failed(String tag, int iteration, int retcode, FLClientStatus curStatus) {
        LOGGER.info(Common.addTag(tag + " failed"));
        LOGGER.info(Common.addTag("========================================================the total response of " + iteration + ": " + curStatus + "======================================================================"));
        flJobResultCallback.onFlJobIterationFinished(flParameter.getFlName(), iteration, retcode);
    }

    public static void main(String[] args) {
        String trainDataset = args[0];
        String vocabFile = args[1];
        String idsFile = args[2];
        String testDataset = args[3];
        String flName = args[4];
        String trainModelPath = args[5];
        String inferModelPath = args[6];
        String clientID = args[7];
        String ip = args[8];
        boolean useSSL = Boolean.parseBoolean(args[9]);
        int port = Integer.parseInt(args[10]);
        int timeWindow = Integer.parseInt(args[11]);
        boolean useElb = Boolean.parseBoolean(args[12]);
        int serverNum = Integer.parseInt(args[13]);
        String task = args[14];
        FLParameter flParameter = FLParameter.getInstance();
        LOGGER.info(Common.addTag("[args] trainDataset: " + trainDataset));
        LOGGER.info(Common.addTag("[args] vocabFile: " + vocabFile));
        LOGGER.info(Common.addTag("[args] idsFile: " + idsFile));
        LOGGER.info(Common.addTag("[args] testDataset: " + testDataset));
        LOGGER.info(Common.addTag("[args] flName: " + flName));
        LOGGER.info(Common.addTag("[args] trainModelPath: " + trainModelPath));
        LOGGER.info(Common.addTag("[args] inferModelPath: " + inferModelPath));
        LOGGER.info(Common.addTag("[args] clientID: " + clientID));
        LOGGER.info(Common.addTag("[args] ip: " + ip));
        LOGGER.info(Common.addTag("[args] useSSL: " + useSSL));
        LOGGER.info(Common.addTag("[args] port: " + port));
        LOGGER.info(Common.addTag("[args] timeWindow: " + timeWindow));
        LOGGER.info(Common.addTag("[args] useElb: " + useElb));
        LOGGER.info(Common.addTag("[args] serverNum: " + serverNum));
        LOGGER.info(Common.addTag("[args] task: " + task));

        flParameter.setClientID(clientID);
        SyncFLJob syncFLJob = new SyncFLJob();
        if (task.equals("train")) {
            flParameter.setTrainDataset(trainDataset);
            flParameter.setFlName(flName);
            flParameter.setTrainModelPath(trainModelPath);
            flParameter.setTestDataset(testDataset);
            flParameter.setInferModelPath(inferModelPath);
            flParameter.setIp(ip);
            flParameter.setUseSSL(useSSL);
            flParameter.setPort(port);
            flParameter.setTimeWindow(timeWindow);
            flParameter.setUseElb(useElb);
            flParameter.setServerNum(serverNum);
            if (ADBERT.equals(flName)) {
                flParameter.setVocabFile(vocabFile);
                flParameter.setIdsFile(idsFile);
            }
            syncFLJob.flJobRun();
        } else if (task.equals("inference")) {
            syncFLJob.modelInference(flName, testDataset, vocabFile, idsFile, inferModelPath);
        } else if (task.equals("getModel")) {
            syncFLJob.getModel(false, 1, ip, port, flName, trainModelPath, inferModelPath, false);
        } else {
            LOGGER.info(Common.addTag("do not do any thing!"));
        }
    }
}
