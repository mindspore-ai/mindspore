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

import com.google.flatbuffers.FlatBufferBuilder;

import com.mindspore.flclient.compression.DecodeExecutor;
import com.mindspore.flclient.model.AlInferBert;
import com.mindspore.flclient.model.AlTrainBert;
import com.mindspore.flclient.model.Client;
import com.mindspore.flclient.model.ClientManager;
import com.mindspore.flclient.model.RunType;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.Status;

import com.mindspore.flclient.model.TrainLenet;
import mindspore.schema.*;

import java.util.List;
import java.util.ArrayList;
import java.util.Date;
import java.util.logging.Logger;

import static com.mindspore.flclient.LocalFLParameter.ALBERT;
import static com.mindspore.flclient.LocalFLParameter.LENET;

/**
 * Define the serialization method, handle the response message returned from server for getModel request.
 *
 * @since 2021-06-30
 */
public class GetModel {
    private static final Logger LOGGER = Logger.getLogger(GetModel.class.toString());
    private static volatile GetModel getModel;

    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private int retCode = ResponseCode.RequestError;

    private GetModel() {
    }

    /**
     * Get the singleton object of the class GetModel.
     *
     * @return the singleton object of the class GetModel.
     */
    public static GetModel getInstance() {
        GetModel localRef = getModel;
        if (localRef == null) {
            synchronized (GetModel.class) {
                localRef = getModel;
                if (localRef == null) {
                    getModel = localRef = new GetModel();
                }
            }
        }
        return localRef;
    }

    public int getRetCode() {
        return retCode;
    }

    /**
     * Get a flatBuffer builder of RequestGetModel.
     *
     * @param name      the model name.
     * @param iteration current iteration of federated learning task.
     * @return the flatBuffer builder of RequestGetModel in byte[] format.
     */
    public byte[] getRequestGetModel(String name, int iteration) {
        if (name == null || name.isEmpty()) {
            LOGGER.severe(Common.addTag("[GetModel] the input parameter of <name> is null or empty, please check!"));
            throw new IllegalArgumentException();
        }
        RequestGetModelBuilder builder = new RequestGetModelBuilder();
        return builder.iteration(iteration).flName(name).time()
                .downloadCompressTypesBuilder(flParameter.getDownloadCompressTypes()).build();
    }

    private FLClientStatus deprecatedParseResponseAlbert(ResponseGetModel responseDataBuf) {
        FLClientStatus status;
        int fmCount = responseDataBuf.featureMapLength();
        if (localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
            LOGGER.info(Common.addTag("[getModel] into <parseResponseAlbert>"));
            ArrayList<FeatureMap> albertFeatureMaps = new ArrayList<FeatureMap>();
            ArrayList<FeatureMap> inferFeatureMaps = new ArrayList<FeatureMap>();
            for (int i = 0; i < fmCount; i++) {
                FeatureMap feature = responseDataBuf.featureMap(i);
                if (feature == null) {
                    LOGGER.severe(Common.addTag("[getModel] the feature returned from server is null"));
                    return FLClientStatus.FAILED;
                }
                String featureName = feature.weightFullname();
                if (localFLParameter.getAlbertWeightName().contains(featureName)) {
                    albertFeatureMaps.add(feature);
                    inferFeatureMaps.add(feature);
                } else if (localFLParameter.getClassifierWeightName().contains(featureName)) {
                    inferFeatureMaps.add(feature);
                } else {
                    continue;
                }
                LOGGER.info(Common.addTag("[getModel] weightFullname: " + feature.weightFullname() + ", weightLength:" +
                        " " + feature.dataLength()));
            }
            status = Common.initSession(flParameter.getInferModelPath());
            if (status == FLClientStatus.FAILED) {
                retCode = ResponseCode.RequestError;
                return status;
            }
            int tag = 0;
            LOGGER.info(Common.addTag("[getModel] ----------------loading weight into inference " +
                    "model-----------------"));
            AlInferBert alInferBert = AlInferBert.getInstance();
            tag = SessionUtil.updateFeatures(alInferBert.getTrainSession(), flParameter.getInferModelPath(),
                    inferFeatureMaps);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[getModel] unsolved error code in <SessionUtil.updateFeatures>"));
                return FLClientStatus.FAILED;
            }
            LOGGER.info(Common.addTag("[getModel] ----------------loading weight into train model-----------------"));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            tag = SessionUtil.updateFeatures(alTrainBert.getTrainSession(), flParameter.getTrainModelPath(),
                    albertFeatureMaps);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[getModel] unsolved error code in <SessionUtil.updateFeatures>"));
                return FLClientStatus.FAILED;
            }
            Common.freeSession();
        } else if (localFLParameter.getServerMod().equals(ServerMod.FEDERATED_LEARNING.toString())) {
            LOGGER.info(Common.addTag("[getModel] into <parseResponseAlbert>"));
            ArrayList<FeatureMap> featureMaps = new ArrayList<FeatureMap>();
            for (int i = 0; i < fmCount; i++) {
                FeatureMap feature = responseDataBuf.featureMap(i);
                if (feature == null) {
                    LOGGER.severe(Common.addTag("[getModel] the feature returned from server is null"));
                    return FLClientStatus.FAILED;
                }
                String featureName = feature.weightFullname();
                featureMaps.add(feature);
                LOGGER.info(Common.addTag("[getModel] weightFullname: " + featureName + ", weightLength: " +
                        feature.dataLength()));
            }
            status = Common.initSession(flParameter.getInferModelPath());
            if (status == FLClientStatus.FAILED) {
                retCode = ResponseCode.RequestError;
                return status;
            }
            int tag = 0;
            LOGGER.info(Common.addTag("[getModel] ----------------loading weight into model-----------------"));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            tag = SessionUtil.updateFeatures(alTrainBert.getTrainSession(), flParameter.getTrainModelPath(),
                    featureMaps);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[getModel] unsolved error code in <SessionUtil.updateFeatures>"));
                return FLClientStatus.FAILED;
            }
            Common.freeSession();
        }
        return FLClientStatus.SUCCESS;
    }

    private FLClientStatus deprecatedParseResponseLenet(ResponseGetModel responseDataBuf) {
        FLClientStatus status;
        int fmCount = responseDataBuf.featureMapLength();
        ArrayList<FeatureMap> featureMaps = new ArrayList<FeatureMap>();
        for (int i = 0; i < fmCount; i++) {
            FeatureMap feature = responseDataBuf.featureMap(i);
            if (feature == null) {
                LOGGER.severe(Common.addTag("[getModel] the feature returned from server is null"));
                return FLClientStatus.FAILED;
            }
            String featureName = feature.weightFullname();
            featureMaps.add(feature);
            LOGGER.info(Common.addTag("[getModel] weightFullname: " + featureName + ", weightLength: " +
                    feature.dataLength()));
        }
        status = Common.initSession(flParameter.getInferModelPath());
        if (status == FLClientStatus.FAILED) {
            retCode = ResponseCode.RequestError;
            return status;
        }
        int tag = 0;
        LOGGER.info(Common.addTag("[getModel] ----------------loading weight into model-----------------"));
        TrainLenet trainLenet = TrainLenet.getInstance();
        tag = SessionUtil.updateFeatures(trainLenet.getTrainSession(), flParameter.getTrainModelPath(), featureMaps);
        if (tag == -1) {
            LOGGER.severe(Common.addTag("[getModel] unsolved error code in <SessionUtil.updateFeatures>"));
            return FLClientStatus.FAILED;
        }
        Common.freeSession();
        return FLClientStatus.SUCCESS;
    }

    private FLClientStatus deprecatedParseFeatures(ResponseGetModel responseDataBuf) {
        FLClientStatus status = FLClientStatus.SUCCESS;
        if (ALBERT.equals(flParameter.getFlName())) {
            LOGGER.info(Common.addTag("[getModel] into <parseResponseAlbert>"));
            status = deprecatedParseResponseAlbert(responseDataBuf);
        } else if (LENET.equals(flParameter.getFlName())) {
            LOGGER.info(Common.addTag("[getModel] into <parseResponseLenet>"));
            status = deprecatedParseResponseLenet(responseDataBuf);
        } else {
            LOGGER.severe(Common.addTag("[getModel] the flName is not valid, only support: lenet, albert"));
            status = FLClientStatus.FAILED;
        }
        return status;
    }

    private List<FeatureMap> parseFeatureMapList(ResponseGetModel responseDataBuf) {
        List<FeatureMap> featureMaps;
        byte compressType = responseDataBuf.downloadCompressType();
        if (responseDataBuf.downloadCompressType() == mindspore.schema.CompressType.NO_COMPRESS) {
            featureMaps = new ArrayList<>();
            for (int i = 0; i < responseDataBuf.featureMapLength(); i++) {
                featureMaps.add(responseDataBuf.featureMap(i));
            }
        } else {
            List<mindspore.schema.CompressFeatureMap> compressFeatureMapList = new ArrayList<>();
            for (int i = 0; i < responseDataBuf.compressFeatureMapLength(); i++) {
                compressFeatureMapList.add(responseDataBuf.compressFeatureMap(i));
            }
            featureMaps = DecodeExecutor.getInstance().deCompressWeight(compressType, compressFeatureMapList);
        }
        return featureMaps;
    }

    private FLClientStatus parseResponseFeatures(ResponseGetModel responseDataBuf) {
        FLClientStatus status;
        Client client = ClientManager.getClient(flParameter.getFlName());
        List<FeatureMap> featureMapList = parseFeatureMapList(responseDataBuf);
        if (featureMapList.size() <= 0) {
            LOGGER.severe(Common.addTag("[getModel] the feature size get from server is zero"));
            retCode = ResponseCode.SystemError;
            return FLClientStatus.FAILED;
        }
        if (localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
            LOGGER.info(Common.addTag("[getModel] parseResponseFeatures by " + localFLParameter.getServerMod()));
            ArrayList<FeatureMap> trainFeatureMaps = new ArrayList<FeatureMap>();
            ArrayList<FeatureMap> inferFeatureMaps = new ArrayList<FeatureMap>();
            for (int i = 0; i < featureMapList.size(); i++) {
                FeatureMap feature = featureMapList.get(i);
                if (feature == null) {
                    LOGGER.severe(Common.addTag("[getModel] the feature returned from server is null"));
                    retCode = ResponseCode.SystemError;
                    return FLClientStatus.FAILED;
                }
                String featureName = feature.weightFullname();
                if (flParameter.getHybridWeightName(RunType.TRAINMODE).contains(featureName)) {
                    trainFeatureMaps.add(feature);
                    LOGGER.info(Common.addTag("[getModel] trainWeightFullname: " + feature.weightFullname() + ", " +
                            "trainWeightLength: " + feature.dataLength()));
                }
                if (flParameter.getHybridWeightName(RunType.INFERMODE).contains(featureName)) {
                    inferFeatureMaps.add(feature);
                    LOGGER.info(Common.addTag("[getModel] inferWeightFullname: " + feature.weightFullname() + ", " +
                            "inferWeightLength: " + feature.dataLength()));
                }
            }
            Status tag;
            LOGGER.info(Common.addTag("[getModel] ----------------loading weight into inference " +
                    "model-----------------"));
            status = Common.initSession(flParameter.getInferModelPath());
            if (status == FLClientStatus.FAILED) {
                retCode = ResponseCode.RequestError;
                return status;
            }
            tag = client.updateFeatures(flParameter.getInferModelPath(), inferFeatureMaps);
            Common.freeSession();
            if (!Status.SUCCESS.equals(tag)) {
                LOGGER.severe(Common.addTag("[getModel] unsolved error code in <Client.updateFeatures>"));
                retCode = ResponseCode.RequestError;
                return FLClientStatus.FAILED;
            }
            LOGGER.info(Common.addTag("[getModel] ----------------loading weight into train model-----------------"));
            status = Common.initSession(flParameter.getTrainModelPath());
            if (status == FLClientStatus.FAILED) {
                retCode = ResponseCode.RequestError;
                return status;
            }
            tag = client.updateFeatures(flParameter.getTrainModelPath(), trainFeatureMaps);
            Common.freeSession();
            if (!Status.SUCCESS.equals(tag)) {
                LOGGER.severe(Common.addTag("[getModel] unsolved error code in <Client.updateFeatures>"));
                retCode = ResponseCode.RequestError;
                return FLClientStatus.FAILED;
            }
        } else if (localFLParameter.getServerMod().equals(ServerMod.FEDERATED_LEARNING.toString())) {
            LOGGER.info(Common.addTag("[getModel] parseResponseFeatures by " + localFLParameter.getServerMod()));
            ArrayList<FeatureMap> featureMaps = new ArrayList<FeatureMap>();
            for (int i = 0; i < featureMapList.size(); i++) {
                FeatureMap feature = featureMapList.get(i);
                if (feature == null) {
                    LOGGER.severe(Common.addTag("[getModel] the feature returned from server is null"));
                    retCode = ResponseCode.SystemError;
                    return FLClientStatus.FAILED;
                }
                String featureName = feature.weightFullname();
                featureMaps.add(feature);
                LOGGER.info(Common.addTag("[getModel] weightFullname: " + featureName + ", " +
                        "weightLength: " + feature.dataLength()));
            }
            Status tag;
            LOGGER.info(Common.addTag("[getModel] ----------------loading weight into model-----------------"));
            status = Common.initSession(flParameter.getTrainModelPath());
            if (status == FLClientStatus.FAILED) {
                retCode = ResponseCode.RequestError;
                return status;
            }
            tag = client.updateFeatures(flParameter.getTrainModelPath(), featureMaps);
            LOGGER.info(Common.addTag("[getModel] ===========free session============="));
            Common.freeSession();
            if (!Status.SUCCESS.equals(tag)) {
                LOGGER.severe(Common.addTag("[getModel] unsolved error code in <Client.updateFeatures>"));
                retCode = ResponseCode.RequestError;
                return FLClientStatus.FAILED;
            }
        }
        return FLClientStatus.SUCCESS;
    }

    /**
     * Handle the response message returned from server.
     *
     * @param responseDataBuf the response message returned from server.
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus doResponse(ResponseGetModel responseDataBuf) {
        retCode = responseDataBuf.retcode();
        LOGGER.info(Common.addTag("[getModel] ==========the response message of getModel is:================"));
        LOGGER.info(Common.addTag("[getModel] ==========retCode: " + retCode));
        LOGGER.info(Common.addTag("[getModel] ==========reason: " + responseDataBuf.reason()));
        LOGGER.info(Common.addTag("[getModel] ==========iteration: " + responseDataBuf.iteration()));
        LOGGER.info(Common.addTag("[getModel] ==========time: " + responseDataBuf.timestamp()));
        FLClientStatus status = FLClientStatus.SUCCESS;
        switch (responseDataBuf.retcode()) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[getModel] getModel response success"));
                if (Common.checkFLName(flParameter.getFlName())) {
                    status = deprecatedParseFeatures(responseDataBuf);
                } else {
                    LOGGER.info(Common.addTag("[getModel] into <parseResponseFeatures>"));
                    status = parseResponseFeatures(responseDataBuf);
                }
                return status;
            case (ResponseCode.SucNotReady):
                LOGGER.info(Common.addTag("[getModel] server is not ready now: need wait and request getModel again"));
                return FLClientStatus.WAIT;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[getModel] out of time: need wait and request startFLJob again"));
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.warning(Common.addTag("[getModel] catch RequestError or SystemError"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[getModel] the return <retCode> from server is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    class RequestGetModelBuilder {
        private FlatBufferBuilder builder;
        private int nameOffset = 0;
        private int iteration = 0;
        private int timeStampOffset = 0;
        private int downloadCompressTypesOffset = 0;

        public RequestGetModelBuilder() {
            builder = new FlatBufferBuilder();
        }

        private RequestGetModelBuilder flName(String name) {
            if (name == null || name.isEmpty()) {
                LOGGER.severe(Common.addTag("[GetModel] the input parameter of <name> is null or empty, please " +
                        "check!"));
                throw new IllegalArgumentException();
            }
            this.nameOffset = this.builder.createString(name);
            return this;
        }

        private RequestGetModelBuilder time() {
            Date date = new Date();
            long time = date.getTime();
            this.timeStampOffset = builder.createString(String.valueOf(time));
            return this;
        }

        private RequestGetModelBuilder iteration(int iteration) {
            this.iteration = iteration;
            return this;
        }

        private RequestGetModelBuilder downloadCompressTypesBuilder(byte[] downloadCompressTypes) {
            if (downloadCompressTypes == null || downloadCompressTypes.length == 0) {
                LOGGER.severe(Common.addTag("[GetModel] the parameter of <downloadCompressTypes> is null or empty," +
                        " please check!"));
                throw new IllegalArgumentException();
            }
            this.downloadCompressTypesOffset = RequestGetModel.createDownloadCompressTypesVector(builder,
                    downloadCompressTypes);
            return this;
        }

        private byte[] build() {
            RequestGetModel.startRequestGetModel(builder);
            RequestGetModel.addFlName(builder, nameOffset);
            RequestGetModel.addIteration(builder, iteration);
            RequestGetModel.addTimestamp(builder, timeStampOffset);
            RequestGetModel.addDownloadCompressTypes(builder, downloadCompressTypesOffset);
            int root = RequestGetModel.endRequestGetModel(builder);
            builder.finish(root);
            return builder.sizedByteArray();
        }
    }
}
