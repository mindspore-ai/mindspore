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

import com.mindspore.flclient.common.FLLoggerGenerater;
import com.mindspore.flclient.model.*;
import com.mindspore.MSTensor;
import com.mindspore.flclient.compression.EncodeExecutor;
import com.mindspore.flclient.compression.CompressWeight;

import mindspore.schema.FeatureMap;
import mindspore.schema.CompressFeatureMap;
import mindspore.schema.RequestUpdateModel;
import mindspore.schema.ResponseCode;
import mindspore.schema.ResponseUpdateModel;

import static com.mindspore.flclient.EncryptLevel.SIGNDS;
import static mindspore.schema.CompressType.NO_COMPRESS;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.security.SecureRandom;

/**
 * Define the serialization method, handle the response message returned from server for updateModel request.
 *
 * @since 2021-06-30
 */
public class UpdateModel {
    private static final Logger LOGGER = FLLoggerGenerater.getModelLogger(UpdateModel.class.toString());
    private static volatile UpdateModel updateModel;

    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private FLClientStatus status;
    private int retCode = ResponseCode.RequestError;

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

    public int getRetCode() {
        return retCode;
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
        boolean isPkiVerify = flParameter.isPkiVerify();
        Client client = ClientManager.getClient(flParameter.getFlName());
        float uploadLoss = client == null ? 0.0f : client.getUploadLoss();
        if (isPkiVerify) {
            Date date = new Date();
            long timestamp = date.getTime();
            String dateTime = String.valueOf(timestamp);
            byte[] signature = CipherClient.signTimeAndIter(dateTime, iteration);
            return builder.flName(flParameter.getFlName()).time(dateTime).id(localFLParameter.getFlID())
                    .featuresMap(secureProtocol, trainDataSize).iteration(iteration)
                    .signData(signature).uploadLoss(uploadLoss).build();
        }
        return builder.flName(flParameter.getFlName()).time("null").id(localFLParameter.getFlID())
                .featuresMap(secureProtocol, trainDataSize).iteration(iteration).uploadLoss(uploadLoss).build();
    }

    /**
     * Handle the response message returned from server.
     *
     * @param response the response message returned from server.
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus doResponse(ResponseUpdateModel response) {
        retCode = response.retcode();
        LOGGER.info("[updateModel] ==========the response message of updateModel is================");
        LOGGER.info("[updateModel] ==========retCode: " + retCode);
        LOGGER.info("[updateModel] ==========reason: " + response.reason());
        LOGGER.info("[updateModel] ==========next request time: " + response.nextReqTime());
        switch (response.retcode()) {
            case (ResponseCode.SUCCEED):
                LOGGER.info("[updateModel] updateModel success");
                return FLClientStatus.SUCCESS;
            case (ResponseCode.OutOfTime):
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.warning("[updateModel] catch RequestError or SystemError");
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe("[updateModel]the return <retCode> from server is invalid: " +
                        response.retcode());
                return FLClientStatus.FAILED;
        }
    }

    class RequestUpdateModelBuilder {
        private RequestUpdateModel requestUM;
        private FlatBufferBuilder builder;
        private int fmOffset = 0;
        private int compFmOffset = 0;
        private int nameOffset = 0;
        private int idOffset = 0;
        private int timestampOffset = 0;
        private int signDataOffset = 0;
        private int sign = 0;
        private int indexArrayOffset = 0;
        private int iteration = 0;
        private byte uploadCompressType = 0;
        private float uploadSparseRate = 0.0f;
        private EncryptLevel encryptLevel = EncryptLevel.NOT_ENCRYPT;
        private float uploadLossOffset = 0.0f;
        private int nameVecOffset = 0;

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
                LOGGER.severe("[updateModel] the parameter of <name> is null or empty, please check!");
                throw new IllegalArgumentException();
            }
            this.nameOffset = this.builder.createString(name);
            return this;
        }

        /**
         * Serialize the element timestamp in RequestUpdateModel.
         *
         * @param setTime current timestamp when the request is sent.
         * @return the RequestUpdateModelBuilder object.
         */
        private RequestUpdateModelBuilder time(String setTime) {
            if (setTime == null || setTime.isEmpty()) {
                LOGGER.severe("[updateModel] the parameter of <setTime> is null or empty, please " +
                        "check!");
                throw new IllegalArgumentException();
            }
            if (setTime.equals("null")) {
                Date date = new Date();
                long time = date.getTime();
                this.timestampOffset = builder.createString(String.valueOf(time));
            } else {
                this.timestampOffset = builder.createString(setTime);
            }
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
                LOGGER.severe("[updateModel] the parameter of <id> is null or empty, please check!");
                throw new IllegalArgumentException();
            }
            this.idOffset = this.builder.createString(id);
            return this;
        }

        abstract class EncrypterBase {
            protected Client client;
            protected SecureProtocol secureProtocol;
            protected ArrayList<String> featureNames;
            protected int trainDataSize;
            protected int curIter = 0;

            public EncrypterBase(Client client, SecureProtocol secureProtocol, int trainDataSize) {
                this.client = client;
                this.secureProtocol = secureProtocol;
                this.trainDataSize = trainDataSize;
                this.featureNames = secureProtocol.getUpdateFeatureName();
            }

            abstract boolean init();

            boolean isEnd() {
                return curIter >= featureNames.size();
            }

            public String getNextFeature() {
                String weightName = featureNames.get(curIter);
                curIter++;
                return weightName;
            }

            abstract public float[] geEncryptWeight(String weightName);
        }

        class NoEncrypter extends EncrypterBase {
            public NoEncrypter(Client client, SecureProtocol secureProtocol, int trainDataSize) {
                super(client, secureProtocol, trainDataSize);
            }

            public boolean init() {
                return true;
            }

            public float[] geEncryptWeight(String weightName) {
                float[] weight = client.getFeature(weightName);
                // here reuse weight to avoid android Background concurrent copying GC freed xxx,  AllocSpace objects xx
                for (int i = 0; i < weight.length; i++) {
                    weight[i] = weight[i] * (float) trainDataSize;
                }
                return weight;
            }
        }

        class PwEncrypter extends EncrypterBase {
            private int maskedLen;

            public PwEncrypter(Client client, SecureProtocol secureProtocol, int trainDataSize) {
                super(client, secureProtocol, trainDataSize);
                this.maskedLen = 0;
            }

            public boolean init() {
                return true;
            }

            public float[] geEncryptWeight(String weightName) {
                float[] weight = client.getFeature(weightName);
                float[] encryptWeight = secureProtocol.pwMaskWeight(trainDataSize, weight, maskedLen);
                maskedLen += maskedLen;
                return encryptWeight;
            }
        }

        class DpEncrypter extends EncrypterBase {
            private double clipFactor;
            private double gaussianSigma;

            public DpEncrypter(Client client, SecureProtocol secureProtocol, int trainDataSize) {
                super(client, secureProtocol, trainDataSize);
            }

            public boolean init() {
                gaussianSigma = secureProtocol.calculateSigma();
                double dpNormClip = secureProtocol.getDpNormClip();
                // calculate l2-norm of all layers' update array
                double updateL2Norm = 0d;
                for (int i = 0; i < featureNames.size(); i++) {
                    String key = featureNames.get(i);
                    float[] data = client.getFeature(key);
                    float[] dataBeforeTrain = client.getPreFeature(key);
                    if (data == null || dataBeforeTrain == null || data.length != dataBeforeTrain.length) {
                        throw new RuntimeException("data of feature size is not same, feature name:" + key);
                    }
                    for (int j = 0; j < data.length; j++) {
                        float updateData = data[j] - dataBeforeTrain[j];
                        updateL2Norm += updateData * updateData;
                    }
                }
                updateL2Norm = Math.sqrt(updateL2Norm);
                if (updateL2Norm == 0) {
                    LOGGER.severe("[Encrypt] updateL2Norm is 0, please check");
                    return false;
                }
                clipFactor = Math.min(1.0, dpNormClip / updateL2Norm);
                return true;
            }

            public float[] geEncryptWeight(String weightName) {
                // clip and add noise
                float[] weight = client.getFeature(weightName);
                float[] weightBeforeTrain = client.getPreFeature(weightName);
                if (weight == null || weightBeforeTrain == null || weight.length != weightBeforeTrain.length) {
                    throw new RuntimeException("data of feature size is not same, feature name:" + weightName);
                }
                // prepare gaussian noise
                // here reuse weight to avoid android Background concurrent copying GC freed xxx,  AllocSpace objects xx
                SecureRandom secureRandom = Common.getSecureRandom();
                for (int j = 0; j < weight.length; j++) {
                    float rawData = weight[j];
                    float rawDataBeforeTrain = weightBeforeTrain[j];
                    float updateData = rawData - rawDataBeforeTrain;
                    // clip
                    updateData *= clipFactor;
                    // add noise
                    double gaussianNoise = secureRandom.nextGaussian() * gaussianSigma;
                    updateData += gaussianNoise;
                    weight[j] = rawDataBeforeTrain + updateData;
                    weight[j] = weight[j] * trainDataSize;
                }
                return weight;
            }
        }


        private EncrypterBase getEncrypter(Client client, SecureProtocol secureProtocol, int trainDataSize) {
            switch (encryptLevel) {
                case PW_ENCRYPT:
                    return new PwEncrypter(client, secureProtocol, trainDataSize);
                case DP_ENCRYPT:
                    return new DpEncrypter(client, secureProtocol, trainDataSize);
                case NOT_ENCRYPT:
                default:
                    return new NoEncrypter(client, secureProtocol, trainDataSize);
            }
        }

        private RequestUpdateModelBuilder featuresMap(SecureProtocol secureProtocol, int trainDataSize) {
            ArrayList<String> updateFeatureName = secureProtocol.getUpdateFeatureName();

            if(encryptLevel == SIGNDS){
                return signDSEncrypt(secureProtocol, updateFeatureName);
            }

            Client client = ClientManager.getClient(flParameter.getFlName());
            EncrypterBase encrypterBase = getEncrypter(client, secureProtocol, trainDataSize);
            encrypterBase.init();
            this.uploadCompressType = localFLParameter.getUploadCompressType();
            if(uploadCompressType == NO_COMPRESS){
                long startTime = System.currentTimeMillis();
                int index = 0;
                int[] fmOffsets = new int[updateFeatureName.size()];
                while (!encrypterBase.isEnd()) {
                    String featureName = encrypterBase.getNextFeature();
                    float[] encryptWeight = encrypterBase.geEncryptWeight(featureName);
                    LOGGER.fine("[updateModel build featuresMap] feature name: " + featureName + " feature " +
                            "size: " + encryptWeight.length);
                    int featureNameOffset = builder.createString(featureName);
                    int weightOffset = FeatureMap.createDataVector(builder, encryptWeight);
                    int featureMapOffset = FeatureMap.createFeatureMap(builder, featureNameOffset, weightOffset);
                    fmOffsets[index] = featureMapOffset;
                    index += 1;
                }
                this.fmOffset = RequestUpdateModel.createFeatureMapVector(builder, fmOffsets);
                long endTime = System.currentTimeMillis();
                LOGGER.info("No compression and encrypt type is:" +
                        encryptLevel + " cost " + (endTime - startTime) + "ms");
                return this;
            }

            long startTime = System.currentTimeMillis();
            int totalMaskLen = 0;
            for (String featureName : updateFeatureName) {
                totalMaskLen += client.getPreFeature(featureName).length;
            }
            boolean[] maskArray = EncodeExecutor.getInstance().constructMaskArray(totalMaskLen);
            int maskedLen = 0;
            int index = 0;
            int[] compFmOffsets = new int[updateFeatureName.size()];
            while (!encrypterBase.isEnd()) {
                String featureName = encrypterBase.getNextFeature();
                float[] encryptWeight = encrypterBase.geEncryptWeight(featureName);
                float[] preWeight = client.getPreFeature(featureName);
                CompressWeight compressWeight = EncodeExecutor.enDiffSparseQuantData(featureName,
                        encryptWeight, preWeight, 8, trainDataSize, maskArray, maskedLen);
                byte[] data = compressWeight.getCompressData();
                float minVal = compressWeight.getMinValue();
                float maxVal = compressWeight.getMaxValue();
                LOGGER.fine("[updateModel build compressWeight] origin size: "
                        + preWeight.length + ", after compress size: " + data.length);
                int featureNameOffset = builder.createString(featureName);
                int weightOffset = CompressFeatureMap.createCompressDataVector(builder, data);
                int featureOffset = CompressFeatureMap.createCompressFeatureMap(builder, featureNameOffset,
                        weightOffset, minVal, maxVal);
                LOGGER.fine("[updateModel Compression] " + featureName +
                        "min_val: " + minVal + ", max_val: " + maxVal);
                compFmOffsets[index] = featureOffset;
                index += 1;
                maskedLen += preWeight.length;
            }
            this.compFmOffset = RequestUpdateModel.createCompressFeatureMapVector(builder, compFmOffsets);
            this.uploadSparseRate = localFLParameter.getUploadSparseRatio();
            this.nameVecOffset = buildNameVecOffset(updateFeatureName);
            long endTime = System.currentTimeMillis();
            LOGGER.info("compression time is " + (endTime - startTime) + "ms, encrypt is " + encryptLevel);
            return this;
        }

        private RequestUpdateModelBuilder signDSEncrypt(SecureProtocol secureProtocol, ArrayList<String> updateFeatureName) {
            long endTime;
            long startTime;
            startTime = System.currentTimeMillis();
            Client client = ClientManager.getClient(flParameter.getFlName());
            // signds alg return indexArray, and package indexArray into flatbuffer.
            SecureRandom secureRandom = Common.getSecureRandom();
            boolean signBool = secureRandom.nextBoolean();
            this.sign = signBool ? 1 : -1;
            int[] indexArray = secureProtocol.signDSModel(client, signBool);
            if (indexArray == null || indexArray.length == 0) {
                LOGGER.severe("[Encrypt] the return fmOffsetsSignDS from <secureProtocol.signDSModel> is " +
                        "null, please check");
                retCode = ResponseCode.RequestError;
                status = FLClientStatus.FAILED;
                throw new IllegalArgumentException();
            }
            this.indexArrayOffset = RequestUpdateModel.createIndexArrayVector(builder, indexArray);

            // only package featureName into flatbuffer.
            int compFeatureSize = updateFeatureName.size();
            int[] fmOffsetsSignds = new int[compFeatureSize];
            for (int i = 0; i < compFeatureSize; i++) {
                String key = updateFeatureName.get(i);
                float[] data = new float[0];
                int featureName = builder.createString(key);
                int weight = FeatureMap.createDataVector(builder, data);
                int featureMap = FeatureMap.createFeatureMap(builder, featureName, weight);
                fmOffsetsSignds[i] = featureMap;
            }
            this.fmOffset = RequestUpdateModel.createFeatureMapVector(builder, fmOffsetsSignds);
            LOGGER.info("[Encrypt] SignDS mask model ok!");
            endTime = System.currentTimeMillis();
            LOGGER.info("signds time is " + (endTime - startTime) + "ms");
            return this;
        }

        private int buildNameVecOffset(ArrayList<String> updateFeatureName) {
            int featureSize = updateFeatureName.size();
            int[] nameVecOffsets = new int[featureSize];
            for (int i = 0; i < featureSize; i++) {
                String key = updateFeatureName.get(i);
                nameVecOffsets[i] = builder.createString(key);
            }
            return RequestUpdateModel.createNameVecVector(builder, nameVecOffsets);
        }

        /**
         * Serialize the element signature in RequestUpdateModel.
         *
         * @param signData the signature Data.
         * @return the RequestUpdateModelBuilder object.
         */
        private RequestUpdateModelBuilder signData(byte[] signData) {
            if (signData == null || signData.length == 0) {
                LOGGER.severe("[updateModel] the parameter of <signData> is null or empty, please " +
                        "check!");
                throw new IllegalArgumentException();
            }
            this.signDataOffset = RequestUpdateModel.createSignatureVector(builder, signData);
            return this;
        }

        /**
         * Serialize the element uploadLoss in RequestUpdateModel.
         *
         * @param upload loss that client train.
         * @return the RequestUpdateModelBuilder object.
         */
        private RequestUpdateModelBuilder uploadLoss(float uploadLoss) {
            this.uploadLossOffset = uploadLoss;
            return this;
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
            RequestUpdateModel.addCompressFeatureMap(builder, this.compFmOffset);
            RequestUpdateModel.addUploadCompressType(builder, this.uploadCompressType);
            RequestUpdateModel.addUploadSparseRate(builder, this.uploadSparseRate);
            RequestUpdateModel.addNameVec(builder, this.nameVecOffset);
            RequestUpdateModel.addFeatureMap(builder, this.fmOffset);
            RequestUpdateModel.addSignature(builder, this.signDataOffset);
            RequestUpdateModel.addUploadLoss(builder, this.uploadLossOffset);
            RequestUpdateModel.addSign(builder, this.sign);
            RequestUpdateModel.addIndexArray(builder, this.indexArrayOffset);
            int root = RequestUpdateModel.endRequestUpdateModel(builder);
            builder.finish(root);
            return builder.sizedByteArray();
        }
    }
}
