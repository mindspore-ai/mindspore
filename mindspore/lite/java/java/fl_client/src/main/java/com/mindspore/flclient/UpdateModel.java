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

import static com.mindspore.flclient.LocalFLParameter.ALBERT;
import static com.mindspore.flclient.LocalFLParameter.LENET;

import com.google.flatbuffers.FlatBufferBuilder;

import com.mindspore.flclient.model.AlTrainBert;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.TrainLenet;

import mindspore.schema.FeatureMap;
import mindspore.schema.RequestUpdateModel;
import mindspore.schema.ResponseCode;
import mindspore.schema.ResponseUpdateModel;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Define the serialization method, handle the response message returned from server for updateModel request.
 *
 * @since 2021-06-30
 */
public class UpdateModel {
    private static final Logger LOGGER = Logger.getLogger(UpdateModel.class.toString());
    private static volatile UpdateModel updateModel;

    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private FLClientStatus status;

    private UpdateModel() {
    }

    /**
     * Get the singleton object of the class UpdateModel.
     *
     * @return the singleton object of the class UpdateModel.
     */
    public static UpdateModel getInstance() {
        UpdateModel localRef = updateModel;
        if (localRef == null) {
            synchronized (UpdateModel.class) {
                localRef = updateModel;
                if (localRef == null) {
                    updateModel = localRef = new UpdateModel();
                }
            }
        }
        return localRef;
    }

    public FLClientStatus getStatus() {
        return status;
    }

    /**
     * Get a flatBuffer builder of RequestUpdateModel.
     *
     * @param iteration      current iteration of federated learning task.
     * @param secureProtocol the object that defines encryption and decryption methods.
     * @param trainDataSize  the size of train date set.
     * @return the flatBuffer builder of RequestUpdateModel in byte[] format.
     */
    public byte[] getRequestUpdateFLJob(int iteration, SecureProtocol secureProtocol, int trainDataSize) {
        RequestUpdateModelBuilder builder = new RequestUpdateModelBuilder(localFLParameter.getEncryptLevel());
        return builder.flName(flParameter.getFlName()).time().id(localFLParameter.getFlID())
                .featuresMap(secureProtocol, trainDataSize).iteration(iteration).build();
    }

    /**
     * Handle the response message returned from server.
     *
     * @param response the response message returned from server.
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus doResponse(ResponseUpdateModel response) {
        LOGGER.info(Common.addTag("[updateModel] ==========updateModel response================"));
        LOGGER.info(Common.addTag("[updateModel] ==========retcode: " + response.retcode()));
        LOGGER.info(Common.addTag("[updateModel] ==========reason: " + response.reason()));
        LOGGER.info(Common.addTag("[updateModel] ==========next request time: " + response.nextReqTime()));
        switch (response.retcode()) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[updateModel] updateModel success"));
                return FLClientStatus.SUCCESS;
            case (ResponseCode.OutOfTime):
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.warning(Common.addTag("[updateModel] catch RequestError or SystemError"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[updateModel]the return <retCode> from server is invalid: " +
                        response.retcode()));
                return FLClientStatus.FAILED;
        }
    }

    class RequestUpdateModelBuilder {
        private RequestUpdateModel requestUM;
        private FlatBufferBuilder builder;
        private int fmOffset = 0;
        private int nameOffset = 0;
        private int idOffset = 0;
        private int timestampOffset = 0;
        private int iteration = 0;
        private EncryptLevel encryptLevel = EncryptLevel.NOT_ENCRYPT;

        private RequestUpdateModelBuilder(EncryptLevel encryptLevel) {
            builder = new FlatBufferBuilder();
            this.encryptLevel = encryptLevel;
        }

        /**
         * Serialize the element flName in RequestUpdateModel.
         *
         * @param name the model name.
         * @return the RequestUpdateModelBuilder object.
         */
        private RequestUpdateModelBuilder flName(String name) {
            if (name == null || name.isEmpty()) {
                LOGGER.severe(Common.addTag("[updateModel] the parameter of <name> is null or empty, please check!"));
                throw new IllegalArgumentException();
            }
            this.nameOffset = this.builder.createString(name);
            return this;
        }

        /**
         * Serialize the element timestamp in RequestUpdateModel.
         *
         * @return the RequestUpdateModelBuilder object.
         */
        private RequestUpdateModelBuilder time() {
            Date date = new Date();
            long time = date.getTime();
            this.timestampOffset = builder.createString(String.valueOf(time));
            return this;
        }

        /**
         * Serialize the element iteration in RequestUpdateModel.
         *
         * @param iteration current iteration of federated learning task.
         * @return the RequestUpdateModelBuilder object.
         */
        private RequestUpdateModelBuilder iteration(int iteration) {
            this.iteration = iteration;
            return this;
        }

        /**
         * Serialize the element fl_id in RequestUpdateModel.
         *
         * @param id a number that uniquely identifies a client.
         * @return the RequestUpdateModelBuilder object.
         */
        private RequestUpdateModelBuilder id(String id) {
            if (id == null || id.isEmpty()) {
                LOGGER.severe(Common.addTag("[updateModel] the parameter of <id> is null or empty, please check!"));
                throw new IllegalArgumentException();
            }
            this.idOffset = this.builder.createString(id);
            return this;
        }

        private RequestUpdateModelBuilder featuresMap(SecureProtocol secureProtocol, int trainDataSize) {
            ArrayList<String> encryptFeatureName = secureProtocol.getEncryptFeatureName();
            switch (encryptLevel) {
                case PW_ENCRYPT:
                    int[] fmOffsetsPW = secureProtocol.pwMaskModel(builder, trainDataSize);
                    if (fmOffsetsPW == null || fmOffsetsPW.length == 0) {
                        LOGGER.severe("[Encrypt] the return fmOffsetsPW from <secureProtocol.pwMaskModel> is " +
                                "null, please check");
                        throw new IllegalArgumentException();
                    }
                    this.fmOffset = RequestUpdateModel.createFeatureMapVector(builder, fmOffsetsPW);
                    LOGGER.info(Common.addTag("[Encrypt] pairwise mask model ok!"));
                    return this;
                case DP_ENCRYPT:
                    int[] fmOffsetsDP = secureProtocol.dpMaskModel(builder, trainDataSize);
                    if (fmOffsetsDP == null || fmOffsetsDP.length == 0) {
                        LOGGER.severe("[Encrypt] the return fmOffsetsDP from <secureProtocol.dpMaskModel> is " +
                                "null, please check");
                        throw new IllegalArgumentException();
                    }
                    this.fmOffset = RequestUpdateModel.createFeatureMapVector(builder, fmOffsetsDP);
                    LOGGER.info(Common.addTag("[Encrypt] DP mask model ok!"));
                    return this;
                case NOT_ENCRYPT:
                default:
                    Map<String, float[]> map = new HashMap<String, float[]>();
                    if (flParameter.getFlName().equals(ALBERT)) {
                        LOGGER.info(Common.addTag("[updateModel] serialize feature map for " +
                                flParameter.getFlName()));
                        AlTrainBert alTrainBert = AlTrainBert.getInstance();
                        map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(alTrainBert.getTrainSession()));
                        if (map.isEmpty()) {
                            LOGGER.severe(Common.addTag("[updateModel] the return map is empty in <SessionUtil" +
                                    ".convertTensorToFeatures>"));
                            status = FLClientStatus.FAILED;
                        }
                    } else if (flParameter.getFlName().equals(LENET)) {
                        LOGGER.info(Common.addTag("[updateModel] serialize feature map for " +
                                flParameter.getFlName()));
                        TrainLenet trainLenet = TrainLenet.getInstance();
                        map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(trainLenet.getTrainSession()));
                        if (map.isEmpty()) {
                            LOGGER.severe(Common.addTag("[updateModel] the return map is empty in <SessionUtil" +
                                    ".convertTensorToFeatures>"));
                            status = FLClientStatus.FAILED;
                        }
                    } else {
                        LOGGER.severe(Common.addTag("[updateModel] the flName is not valid"));
                        throw new IllegalArgumentException();
                    }
                    int featureSize = encryptFeatureName.size();
                    int[] fmOffsets = new int[featureSize];
                    for (int i = 0; i < featureSize; i++) {
                        String key = encryptFeatureName.get(i);
                        float[] data = map.get(key);
                        LOGGER.info(Common.addTag("[updateModel build featuresMap] feature name: " + key + " feature " +
                                "size: " + data.length));
                        for (int j = 0; j < data.length; j++) {
                            data[j] = data[j] * trainDataSize;
                        }
                        int featureName = builder.createString(key);
                        int weight = FeatureMap.createDataVector(builder, data);
                        int featureMap = FeatureMap.createFeatureMap(builder, featureName, weight);
                        fmOffsets[i] = featureMap;
                    }
                    this.fmOffset = RequestUpdateModel.createFeatureMapVector(builder, fmOffsets);
                    return this;
            }
        }

        /**
         * Create a flatBuffer builder of RequestUpdateModel.
         *
         * @return the flatBuffer builder of RequestUpdateModel in byte[] format.
         */
        private byte[] build() {
            RequestUpdateModel.startRequestUpdateModel(this.builder);
            RequestUpdateModel.addFlName(builder, nameOffset);
            RequestUpdateModel.addFlId(this.builder, idOffset);
            RequestUpdateModel.addTimestamp(builder, this.timestampOffset);
            RequestUpdateModel.addIteration(builder, this.iteration);
            RequestUpdateModel.addFeatureMap(builder, this.fmOffset);
            int root = RequestUpdateModel.endRequestUpdateModel(builder);
            builder.finish(root);
            return builder.sizedByteArray();
        }
    }
}
