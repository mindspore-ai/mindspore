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
import com.mindspore.flclient.model.AdInferBert;
import com.mindspore.flclient.model.AdTrainBert;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.TrainLenet;
import mindspore.schema.FeatureMap;
import mindspore.schema.RequestFLJob;
import mindspore.schema.ResponseCode;
import mindspore.schema.ResponseFLJob;

import java.util.ArrayList;
import java.util.logging.Logger;

public class StartFLJob {
    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private static final Logger LOGGER = Logger.getLogger(StartFLJob.class.toString());

    class RequestStartFLJobBuilder {
        private RequestFLJob requestFLJob;
        private FlatBufferBuilder builder;
        private int nameOffset = 0;
        private int iteration = 0;
        private int dataSize = 0;
        private int timestampOffset = 0;
        private int idOffset = 0;

        public RequestStartFLJobBuilder() {
            builder = new FlatBufferBuilder();
        }

        public RequestStartFLJobBuilder flName(String name) {
            this.nameOffset = this.builder.createString(name);
            return this;
        }

        public RequestStartFLJobBuilder id(String id) {
            this.idOffset = this.builder.createString(id);
            return this;
        }

        public RequestStartFLJobBuilder time(long timestamp) {
            this.timestampOffset = builder.createString(String.valueOf(timestamp));
            return this;
        }

        public RequestStartFLJobBuilder dataSize(int dataSize) {
            // temp code need confirm
            this.dataSize = dataSize;
            LOGGER.info(Common.addTag("[startFLJob] the train data size: " + dataSize));
            return this;
        }

        public RequestStartFLJobBuilder iteration(int iteration) {
            this.iteration = iteration;
            return this;
        }

        public byte[] build() {
            int root = RequestFLJob.createRequestFLJob(this.builder, this.nameOffset, this.idOffset, this.iteration,
                    this.dataSize, this.timestampOffset);
            builder.finish(root);
            return builder.sizedByteArray();
        }
    }

    private static StartFLJob startFLJob;

    private FLClientStatus status;

    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private int featureSize;
    private String nextRequestTime;
    private ArrayList<String> encryptFeatureName = new ArrayList<String>();

    private StartFLJob() {

    }

    public static StartFLJob getInstance() {
        if (startFLJob == null) {
            startFLJob = new StartFLJob();
        }
        return startFLJob;
    }

    public String getNextRequestTime() {
        return nextRequestTime;
    }

    public byte[] getRequestStartFLJob(int dataSize, int iteration, long time) {
        RequestStartFLJobBuilder builder = new RequestStartFLJobBuilder();
        return builder.flName(flParameter.getFlName())
                .time(time)
                .id(localFLParameter.getFlID())
                .dataSize(dataSize)
                .iteration(iteration)
                .build();
    }

    public int getFeatureSize() {
        return featureSize;
    }

    public ArrayList<String> getEncryptFeatureName() {
        return encryptFeatureName;
    }

    private FLClientStatus parseResponseAdbert(ResponseFLJob flJob) {
        int fmCount = flJob.featureMapLength();
        ArrayList<FeatureMap> albertFeatureMaps = new ArrayList<FeatureMap>();
        ArrayList<FeatureMap> inferFeatureMaps = new ArrayList<FeatureMap>();
        encryptFeatureName.clear();
        if (fmCount <= 0) {
            LOGGER.severe(Common.addTag("[startFLJob] the feature size get from server is zero"));
            return FLClientStatus.FAILED;
        }
        for (int i = 0; i < fmCount; i++) {
            FeatureMap feature = flJob.featureMap(i);
            String featureName = feature.weightFullname();
            if (localFLParameter.getAlbertWeightName().contains(featureName)) {
                albertFeatureMaps.add(feature);
                inferFeatureMaps.add(feature);
                featureSize += feature.dataLength();
                encryptFeatureName.add(feature.weightFullname());
            } else if (localFLParameter.getClassifierWeightName().contains(featureName)) {
                inferFeatureMaps.add(feature);
            } else {
                continue;
            }
            LOGGER.info(Common.addTag("[startFLJob] weightFullname: " + feature.weightFullname() + ", weightLength: " + feature.dataLength()));
        }
        int tag = 0;
        LOGGER.info(Common.addTag("[startFLJob] ----------------loading weight into inference model-----------------"));
        AdInferBert adInferBert = AdInferBert.getInstance();
        tag = SessionUtil.updateFeatures(adInferBert.getTrainSession(), flParameter.getInferModelPath(), inferFeatureMaps);
        if (tag == -1) {
            LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in <SessionUtil.updateFeatures>"));
            return FLClientStatus.FAILED;
        }
        LOGGER.info(Common.addTag("[startFLJob] ----------------loading weight into train model-----------------"));
        AdTrainBert adTrainBert = AdTrainBert.getInstance();
        tag = SessionUtil.updateFeatures(adTrainBert.getTrainSession(), flParameter.getTrainModelPath(), albertFeatureMaps);
        if (tag == -1) {
            LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in <SessionUtil.updateFeatures>"));
            return FLClientStatus.FAILED;
        }
        return FLClientStatus.SUCCESS;
    }

    private FLClientStatus parseResponseLenet(ResponseFLJob flJob) {
        int fmCount = flJob.featureMapLength();
        ArrayList<FeatureMap> featureMaps = new ArrayList<FeatureMap>();
        encryptFeatureName.clear();
        for (int i = 0; i < fmCount; i++) {
            FeatureMap feature = flJob.featureMap(i);
            String featureName = feature.weightFullname();
            featureMaps.add(feature);
            featureSize += feature.dataLength();
            encryptFeatureName.add(featureName);
            LOGGER.info(Common.addTag("[startFLJob] weightFullname: " + feature.weightFullname() + ", weightLength: " + feature.dataLength()));
        }
        int tag = 0;
        LOGGER.info(Common.addTag("[startFLJob] ----------------loading weight into model-----------------"));
        TrainLenet trainLenet = TrainLenet.getInstance();
        tag = SessionUtil.updateFeatures(trainLenet.getTrainSession(), flParameter.getTrainModelPath(), featureMaps);
        if (tag == -1) {
            LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in <SessionUtil.updateFeatures>"));
            return FLClientStatus.FAILED;
        }
        return FLClientStatus.SUCCESS;
    }

    public FLClientStatus doResponse(ResponseFLJob flJob) {
        LOGGER.info(Common.addTag("[startFLJob] return code: " + flJob.retcode()));
        LOGGER.info(Common.addTag("[startFLJob] reason: " + flJob.reason()));
        LOGGER.info(Common.addTag("[startFLJob] iteration: " + flJob.iteration()));
        LOGGER.info(Common.addTag("[startFLJob] is selected: " + flJob.isSelected()));
        LOGGER.info(Common.addTag("[startFLJob] next request time: " + flJob.nextReqTime()));
        nextRequestTime = flJob.nextReqTime();
        LOGGER.info(Common.addTag("[startFLJob] timestamp: " + flJob.timestamp()));
        int retcode = flJob.retcode();

        switch (retcode) {
            case (ResponseCode.SUCCEED):
                localFLParameter.setServerMod(flJob.flPlanConfig().serverMode());
                if (localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
                    LOGGER.info(Common.addTag("[startFLJob] into <parseResponseAdbert>"));
                    parseResponseAdbert(flJob);
                } else if (localFLParameter.getServerMod().equals(ServerMod.FEDERATED_LEARNING.toString())) {
                    LOGGER.info(Common.addTag("[startFLJob] into <parseResponseLenet>"));
                    parseResponseLenet(flJob);
                }
                return FLClientStatus.SUCCESS;
            case (ResponseCode.OutOfTime):
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[startFLJob] catch RequestError or SystemError"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[startFLJob] the return <retcode> from server is invalid: " + retcode));
                return FLClientStatus.FAILED;
        }
    }

    public FLClientStatus getStatus() {
        return this.status;
    }
}
