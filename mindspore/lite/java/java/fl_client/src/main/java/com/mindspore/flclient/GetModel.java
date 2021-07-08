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

import com.google.flatbuffers.FlatBufferBuilder;
import com.mindspore.flclient.model.AlInferBert;
import com.mindspore.flclient.model.AlTrainBert;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.TrainLenet;
import mindspore.schema.FeatureMap;
import mindspore.schema.RequestGetModel;
import mindspore.schema.ResponseCode;
import mindspore.schema.ResponseGetModel;

import java.util.ArrayList;
import java.util.Date;
import java.util.logging.Logger;

import static com.mindspore.flclient.LocalFLParameter.ALBERT;
import static com.mindspore.flclient.LocalFLParameter.LENET;

public class GetModel {
    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    class RequestGetModelBuilder {
        private FlatBufferBuilder builder;
        private int nameOffset = 0;
        private int iteration = 0;
        private int timeStampOffset = 0;

        public RequestGetModelBuilder() {
            builder = new FlatBufferBuilder();
        }

        public RequestGetModelBuilder flName(String name) {
            this.nameOffset = this.builder.createString(name);
            return this;
        }

        public RequestGetModelBuilder time() {
            Date date = new Date();
            long time = date.getTime();
            this.timeStampOffset = builder.createString(String.valueOf(time));
            return this;
        }

        public RequestGetModelBuilder iteration(int iteration) {
            this.iteration = iteration;
            return this;
        }

        public byte[] build() {
            int root = RequestGetModel.createRequestGetModel(this.builder, nameOffset, iteration, timeStampOffset);
            builder.finish(root);
            return builder.sizedByteArray();
        }
    }

    private static final Logger LOGGER = Logger.getLogger(GetModel.class.toString());
    private static GetModel getModel;

    private GetModel() {
    }

    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();

    public static GetModel getInstance() {
        if (getModel == null) {
            getModel = new GetModel();
        }
        return getModel;
    }

    public byte[] getRequestGetModel(String name, int iteration) {
        RequestGetModelBuilder builder = new RequestGetModelBuilder();
        return builder.iteration(iteration).flName(name).time().build();
    }


    private FLClientStatus parseResponseAlbert(ResponseGetModel responseDataBuf) {
        int fmCount = responseDataBuf.featureMapLength();
        if (localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
            LOGGER.info(Common.addTag("[getModel] into <parseResponseAdbert>"));
            ArrayList<FeatureMap> albertFeatureMaps = new ArrayList<FeatureMap>();
            ArrayList<FeatureMap> inferFeatureMaps = new ArrayList<FeatureMap>();
            for (int i = 0; i < fmCount; i++) {
                FeatureMap feature = responseDataBuf.featureMap(i);
                String featureName = feature.weightFullname();
                if (localFLParameter.getAlbertWeightName().contains(featureName)) {
                    albertFeatureMaps.add(feature);
                    inferFeatureMaps.add(feature);
                } else if (localFLParameter.getClassifierWeightName().contains(featureName)) {
                    inferFeatureMaps.add(feature);
                } else {
                    continue;
                }
                LOGGER.info(Common.addTag("[getModel] weightFullname: " + feature.weightFullname() + ", weightLength: " + feature.dataLength()));
            }
            int tag = 0;
            LOGGER.info(Common.addTag("[getModel] ----------------loading weight into inference model-----------------"));
            AlInferBert alInferBert = AlInferBert.getInstance();
            tag = SessionUtil.updateFeatures(alInferBert.getTrainSession(), flParameter.getInferModelPath(), inferFeatureMaps);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[getModel] unsolved error code in <SessionUtil.updateFeatures>"));
                return FLClientStatus.FAILED;
            }
            LOGGER.info(Common.addTag("[getModel] ----------------loading weight into train model-----------------"));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            tag = SessionUtil.updateFeatures(alTrainBert.getTrainSession(), flParameter.getTrainModelPath(), albertFeatureMaps);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[getModel] unsolved error code in <SessionUtil.updateFeatures>"));
                return FLClientStatus.FAILED;
            }
        } else if (localFLParameter.getServerMod().equals(ServerMod.FEDERATED_LEARNING.toString())) {
            LOGGER.info(Common.addTag("[getModel] into <parseResponseLenet>"));
            ArrayList<FeatureMap> featureMaps = new ArrayList<FeatureMap>();
            for (int i = 0; i < fmCount; i++) {
                FeatureMap feature = responseDataBuf.featureMap(i);
                String featureName = feature.weightFullname();
                featureMaps.add(feature);
                LOGGER.info(Common.addTag("[getModel] weightFullname: " + featureName + ", weightLength: " + feature.dataLength()));
            }
            int tag = 0;
            LOGGER.info(Common.addTag("[getModel] ----------------loading weight into model-----------------"));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            tag = SessionUtil.updateFeatures(alTrainBert.getTrainSession(), flParameter.getTrainModelPath(), featureMaps);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[getModel] unsolved error code in <SessionUtil.updateFeatures>"));
                return FLClientStatus.FAILED;
            }
        }
        return FLClientStatus.SUCCESS;
    }

    private FLClientStatus parseResponseLenet(ResponseGetModel responseDataBuf) {
        int fmCount = responseDataBuf.featureMapLength();
        ArrayList<FeatureMap> featureMaps = new ArrayList<FeatureMap>();
        for (int i = 0; i < fmCount; i++) {
            FeatureMap feature = responseDataBuf.featureMap(i);
            String featureName = feature.weightFullname();
            featureMaps.add(feature);
            LOGGER.info(Common.addTag("[getModel] weightFullname: " + featureName + ", weightLength: " + feature.dataLength()));
        }
        int tag = 0;
        LOGGER.info(Common.addTag("[getModel] ----------------loading weight into model-----------------"));
        TrainLenet trainLenet = TrainLenet.getInstance();
        tag = SessionUtil.updateFeatures(trainLenet.getTrainSession(), flParameter.getTrainModelPath(), featureMaps);
        if (tag == -1) {
            LOGGER.severe(Common.addTag("[getModel] unsolved error code in <SessionUtil.updateFeatures>"));
            return FLClientStatus.FAILED;
        }
        return FLClientStatus.SUCCESS;
    }


    public FLClientStatus doResponse(ResponseGetModel responseDataBuf) {
        LOGGER.info(Common.addTag("[getModel] ==========get model content is:================"));
        LOGGER.info(Common.addTag("[getModel] ==========retCode: " + responseDataBuf.retcode()));
        LOGGER.info(Common.addTag("[getModel] ==========reason: " + responseDataBuf.reason()));
        LOGGER.info(Common.addTag("[getModel] ==========iteration: " + responseDataBuf.iteration()));
        LOGGER.info(Common.addTag("[getModel] ==========time: " + responseDataBuf.timestamp()));
        FLClientStatus status = FLClientStatus.SUCCESS;
        int retCode = responseDataBuf.retcode();
        switch (retCode) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[getModel] getModel response success"));

                if (ALBERT.equals(flParameter.getFlName())) {
                    LOGGER.info(Common.addTag("[getModel] into <parseResponseAlbert>"));
                    status = parseResponseAlbert(responseDataBuf);
                } else if (LENET.equals(flParameter.getFlName())) {
                    LOGGER.info(Common.addTag("[getModel] into <parseResponseLenet>"));
                    status = parseResponseLenet(responseDataBuf);
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

}
