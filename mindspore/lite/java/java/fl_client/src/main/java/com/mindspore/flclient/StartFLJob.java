/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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
import com.mindspore.flclient.compression.DecodeExecutor;
import com.mindspore.flclient.model.Client;
import com.mindspore.flclient.model.ClientManager;
import com.mindspore.flclient.model.RunType;
import com.mindspore.flclient.model.Status;
import com.mindspore.flclient.pki.PkiBean;
import com.mindspore.flclient.pki.PkiUtil;

import mindspore.schema.*;
import mindspore.schema.FLPlan;
import mindspore.schema.FeatureMap;
import mindspore.schema.RequestFLJob;
import mindspore.schema.ResponseCode;
import mindspore.schema.ResponseFLJob;

import java.io.IOException;
import java.security.cert.Certificate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.logging.Logger;

/**
 * StartFLJob
 *
 * @since 2021-08-25
 */
public class StartFLJob {
    private static final Logger LOGGER = FLLoggerGenerater.getModelLogger(StartFLJob.class.toString());
    private static volatile StartFLJob startFLJob;

    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();

    private int featureSize;
    private String nextRequestTime;
    private ArrayList<String> updateFeatureName = new ArrayList<String>();
    private int retCode = ResponseCode.RequestError;
    private float lr = (float) 0.1;
    private int batchSize;

    private StartFLJob() {
    }

    /**
     * getInstance of StartFLJob
     *
     * @return StartFLJob instance
     */
    public static StartFLJob getInstance() {
        StartFLJob localRef = startFLJob;
        if (localRef == null) {
            synchronized (StartFLJob.class) {
                localRef = startFLJob;
                if (localRef == null) {
                    startFLJob = localRef = new StartFLJob();
                }
            }
        }
        return localRef;
    }

    public String getNextRequestTime() {
        return nextRequestTime;
    }

    public int getRetCode() {
        return retCode;
    }

    /**
     * get request start FLJob
     *
     * @param dataSize  dataSize
     * @param iteration iteration
     * @param time      time
     * @param pkiBean   pki bean
     * @return byte[] data
     */
    public byte[] getRequestStartFLJob(int dataSize, int iteration, long time, PkiBean pkiBean) {
        RequestStartFLJobBuilder builder = new RequestStartFLJobBuilder();

        if (flParameter.isPkiVerify()) {
            if (pkiBean == null) {
                LOGGER.severe("[startFLJob] the parameter of <pkiBean> is null, please check!");
                throw new IllegalArgumentException();
            }
            return builder.flName(flParameter.getFlName())
                    .time(time)
                    .id(localFLParameter.getFlID())
                    .dataSize(dataSize)
                    .iteration(iteration)
                    .signData(pkiBean.getSignData())
                    .certificateChain(pkiBean.getCertificates())
                    .downloadCompressTypesBuilder(flParameter.getDownloadCompressTypes())
                    .build();
        }
        return builder.flName(flParameter.getFlName())
                .time(time)
                .id(localFLParameter.getFlID())
                .dataSize(dataSize)
                .iteration(iteration)
                .downloadCompressTypesBuilder(flParameter.getDownloadCompressTypes())
                .build();
    }

    public int getFeatureSize() {
        return featureSize;
    }

    public ArrayList<String> getUpdateFeatureName() {
        return updateFeatureName;
    }

    abstract class FeatureGenerator {
        protected ResponseFLJob responseDataBuf;
        protected int curPos = 0;
        protected int size = 0;

        public FeatureGenerator(ResponseFLJob responseDataBuf) {
            this.responseDataBuf = responseDataBuf;
        }

        public abstract FeatureMap next();

        public boolean isEnd() {
            return curPos >= size;
        }
    }

    class NormalFeatureGenerator extends FeatureGenerator {
        public NormalFeatureGenerator(ResponseFLJob responseDataBuf) {
            super(responseDataBuf);
            this.size = responseDataBuf.featureMapLength();
        }

        @Override
        public FeatureMap next() {
            if (curPos >= size) {
                return null;
            }
            int pre = curPos++;
            return responseDataBuf.featureMap(pre);
        }
    }

    class QuatFeatureGenerator extends FeatureGenerator {
        public QuatFeatureGenerator(ResponseFLJob responseDataBuf) {
            super(responseDataBuf);
            this.size = responseDataBuf.compressFeatureMapLength();
        }

        @Override
        public FeatureMap next() {
            if (curPos >= size) {
                return null;
            }
            int pre = curPos++;
            CompressFeatureMap cmpfeatureMap = responseDataBuf.compressFeatureMap(pre);
            return DecodeExecutor.quantDeCompress(cmpfeatureMap);
        }
    }


    private FeatureGenerator FeatureGeneratorCtr(ResponseFLJob responseDataBuf) {
        byte compressType = responseDataBuf.downloadCompressType();
        switch (compressType) {
            case CompressType.NO_COMPRESS:
                return new NormalFeatureGenerator(responseDataBuf);
            case CompressType.QUANT:
                return new QuatFeatureGenerator(responseDataBuf);
            default:
                LOGGER.severe("[FeatureGeneratorCtr] Unsupported compress type:" + compressType);
                return null;
        }
    }

    private FLClientStatus updateFeatureForFederated(Client client, FeatureGenerator featureGenerator) {
        FLClientStatus result = FLClientStatus.SUCCESS;
        Status status = Status.SUCCESS;
        updateFeatureName.clear();
        featureSize = 0;
        LOGGER.info("[startFLJob] set <batch size> for client: " + batchSize);
        client.setBatchSize(batchSize);
        while (!featureGenerator.isEnd()) {
            FeatureMap featureMap = featureGenerator.next();
            featureSize += featureMap.dataLength();
            updateFeatureName.add(featureMap.weightFullname());
            status = client.updateFeature(featureMap, true);
            if (status != Status.SUCCESS) {
                LOGGER.severe("[updateFeatureForFederated] Update feature failed.");
                break;
            }
        }

        return status == Status.SUCCESS ? FLClientStatus.SUCCESS : FLClientStatus.FAILED;
    }

    private FLClientStatus updateFeatureForHybrid(Client client, FeatureGenerator featureGenerator) {
        FLClientStatus result = FLClientStatus.SUCCESS;
        Status status = Status.SUCCESS;
        updateFeatureName.clear();
        featureSize = 0;
        LOGGER.info("[startFLJob] set <batch size> for client: " + batchSize);
        client.setBatchSize(batchSize);
        while (!featureGenerator.isEnd()) {
            FeatureMap featureMap = featureGenerator.next();
            if (flParameter.getHybridWeightName(RunType.TRAINMODE).contains(featureMap.weightFullname())) {
                featureSize += featureMap.dataLength();
                updateFeatureName.add(featureMap.weightFullname());
                status = client.updateFeature(featureMap, true);
            }
            if (status != Status.SUCCESS) {
                LOGGER.severe("[updateFeatureForFederated] Update feature failed.");
                break;
            }
            if (flParameter.getHybridWeightName(RunType.INFERMODE).contains(featureMap.weightFullname())) {
                status = client.updateFeature(featureMap, false);
            }
            if (status != Status.SUCCESS) {
                LOGGER.severe("[updateFeatureForFederated] Update feature failed.");
                break;
            }
        }
        return status == Status.SUCCESS ? FLClientStatus.SUCCESS : FLClientStatus.FAILED;
    }


    private FLClientStatus parseResponseFeatures(ResponseFLJob flJob) {
        FLClientStatus status = FLClientStatus.FAILED;
        Client client = ClientManager.getClient(flParameter.getFlName());
        FeatureGenerator featureGenerator = FeatureGeneratorCtr(flJob);
        if (localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
            LOGGER.info("[startFLJob] parseResponseFeatures by " + localFLParameter.getServerMod());
            status = updateFeatureForHybrid(client, featureGenerator);
        } else if (localFLParameter.getServerMod().equals(ServerMod.FEDERATED_LEARNING.toString())) {
            LOGGER.info("[startFLJob] parseResponseFeatures by " + localFLParameter.getServerMod());
            status = updateFeatureForFederated(client, featureGenerator);
        }
        return status;
    }

    /**
     * response res
     *
     * @param flJob ResponseFLJob
     * @return FLClientStatus
     */
    public FLClientStatus doResponse(ResponseFLJob flJob) {
        if (flJob == null) {
            LOGGER.severe("[startFLJob] the input parameter flJob is null");
            retCode = ResponseCode.SystemError;
            return FLClientStatus.FAILED;
        }
        FLPlan flPlanConfig = flJob.flPlanConfig();
        if (flPlanConfig == null) {
            LOGGER.severe("[startFLJob] the flPlanConfig is null");
            retCode = ResponseCode.SystemError;
            return FLClientStatus.FAILED;
        }

        retCode = flJob.retcode();
        LOGGER.info("[startFLJob] ==========the response message of startFLJob is:================");
        LOGGER.info("[startFLJob] return retCode: " + retCode);
        LOGGER.info("[startFLJob] reason: " + flJob.reason());
        LOGGER.info("[startFLJob] iteration: " + flJob.iteration());
        LOGGER.info("[startFLJob] is selected: " + flJob.isSelected());
        LOGGER.info("[startFLJob] next request time: " + flJob.nextReqTime());
        nextRequestTime = flJob.nextReqTime();
        LOGGER.info("[startFLJob] timestamp: " + flJob.timestamp());
        FLClientStatus status;
        int responseRetCode = flJob.retcode();

        switch (responseRetCode) {
            case (ResponseCode.SUCCEED):
                if (flJob.downloadCompressType() == CompressType.NO_COMPRESS && flJob.featureMapLength() <= 0) {
                    LOGGER.warning("[startFLJob] the feature size get from server is zero");
                    retCode = ResponseCode.SystemError;
                    return FLClientStatus.FAILED;
                }
                localFLParameter.setServerMod(flPlanConfig.serverMode());
                if (flPlanConfig.lr() != 0) {
                    lr = flPlanConfig.lr();
                } else {
                    LOGGER.info("[startFLJob] the GlobalParameter <lr> from server: " + lr + " is not " +
                            "valid, " +
                            "will use the default value 0.1");
                }
                localFLParameter.setLr(lr);
                batchSize = flPlanConfig.miniBatch();
                LOGGER.info("[startFLJob] into <parseResponseFeatures>");
                status = parseResponseFeatures(flJob);
                return status;
            case (ResponseCode.OutOfTime):
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info("[startFLJob] catch RequestError or SystemError");
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe("[startFLJob] the return <retCode> from server is invalid: " + retCode);
                return FLClientStatus.FAILED;
        }
    }

    class RequestStartFLJobBuilder {
        private RequestFLJob requestFLJob;
        private FlatBufferBuilder builder;
        private int nameOffset = 0;
        private int iteration = 0;
        private int dataSize = 0;
        private int timestampOffset = 0;
        private int idOffset = 0;
        private int signDataOffset = 0;
        private int keyAttestationOffset = 0;
        private int equipCertOffset = 0;
        private int equipCACertOffset = 0;
        private int rootCertOffset = 0;
        private int downloadCompressTypesOffset = 0;

        public RequestStartFLJobBuilder() {
            builder = new FlatBufferBuilder();
        }

        /**
         * set flName
         *
         * @param name String
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder flName(String name) {
            if (name == null || name.isEmpty()) {
                LOGGER.severe("[startFLJob] the parameter of <name> is null or empty, please check!");
                throw new IllegalArgumentException();
            }
            this.nameOffset = this.builder.createString(name);
            return this;
        }

        /**
         * set id
         *
         * @param id String
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder id(String id) {
            if (id == null || id.isEmpty()) {
                LOGGER.severe("[startFLJob] the parameter of <id> is null or empty, please check!");
                throw new IllegalArgumentException();
            }
            this.idOffset = this.builder.createString(id);
            return this;
        }

        /**
         * set time
         *
         * @param timestamp long
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder time(long timestamp) {
            this.timestampOffset = builder.createString(String.valueOf(timestamp));
            return this;
        }

        /**
         * set dataSize
         *
         * @param dataSize int
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder dataSize(int dataSize) {
            // temp code need confirm
            this.dataSize = dataSize;
            LOGGER.info("[startFLJob] the train data size: " + dataSize);
            return this;
        }

        /**
         * set iteration
         *
         * @param iteration iteration
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder iteration(int iteration) {
            this.iteration = iteration;
            return this;
        }

        /**
         * signData
         *
         * @param signData byte[]
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder signData(byte[] signData) {
            if (signData == null || signData.length == 0) {
                LOGGER.severe(
                        "[startFLJob] the parameter of <signData> is null or empty, please check!");
                throw new IllegalArgumentException();
            }
            this.signDataOffset = RequestFLJob.createSignDataVector(builder, signData);
            return this;
        }

        /**
         * set certificateChain
         *
         * @param certificates Certificate array
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder certificateChain(Certificate[] certificates) {
            if (certificates == null || certificates.length < 4) {
                LOGGER.severe("[startFLJob] the parameter of <certificates> is null or the length "
                        + "is not valid (should be >= 4), please check!");
                throw new IllegalArgumentException();
            }
            try {
                String keyAttestationPem = PkiUtil.getPemFormat(certificates[0]);
                String equipCertPem = PkiUtil.getPemFormat(certificates[1]);
                String equipCACertPem = PkiUtil.getPemFormat(certificates[2]);
                String rootCertPem = PkiUtil.getPemFormat(certificates[3]);

                this.keyAttestationOffset = this.builder.createString(keyAttestationPem);
                this.equipCertOffset = this.builder.createString(equipCertPem);
                this.equipCACertOffset = this.builder.createString(equipCACertPem);
                this.rootCertOffset = this.builder.createString(rootCertPem);
            } catch (IOException e) {
                LOGGER.severe("[StartFLJob] catch IOException in certificateChain: " + e.getMessage());
            }
            return this;
        }

        private RequestStartFLJobBuilder downloadCompressTypesBuilder(byte[] downloadCompressTypes) {
            if (downloadCompressTypes == null || downloadCompressTypes.length == 0) {
                LOGGER.severe("[StartFLJob] the parameter of <downloadCompressTypes> is null or empty," +
                        " please check!");
                throw new IllegalArgumentException();
            }
            this.downloadCompressTypesOffset = RequestFLJob.createDownloadCompressTypesVector(builder,
                    downloadCompressTypes);
            return this;
        }

        /**
         * build protobuffer
         *
         * @return byte[] data
         */
        public byte[] build() {
            RequestFLJob.startRequestFLJob(this.builder);
            RequestFLJob.addFlName(builder, nameOffset);
            RequestFLJob.addFlId(builder, idOffset);
            RequestFLJob.addIteration(builder, iteration);
            RequestFLJob.addDataSize(builder, dataSize);
            RequestFLJob.addTimestamp(builder, timestampOffset);
            RequestFLJob.addSignData(builder, signDataOffset);
            RequestFLJob.addRootCert(builder, rootCertOffset);
            RequestFLJob.addEquipCaCert(builder, equipCACertOffset);
            RequestFLJob.addEquipCert(builder, equipCertOffset);
            RequestFLJob.addKeyAttestation(builder, keyAttestationOffset);
            RequestFLJob.addDownloadCompressTypes(builder, downloadCompressTypesOffset);
            int root = RequestFLJob.endRequestFLJob(builder);
            builder.finish(root);
            return builder.sizedByteArray();
        }
    }
}
