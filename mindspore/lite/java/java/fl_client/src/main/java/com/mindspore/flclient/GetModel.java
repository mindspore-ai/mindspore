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
import com.mindspore.flclient.compression.DecodeExecutor;
import com.mindspore.flclient.model.Client;
import com.mindspore.flclient.model.ClientManager;
import com.mindspore.flclient.model.RunType;
import com.mindspore.flclient.model.Status;

import mindspore.schema.*;

import java.util.List;
import java.util.ArrayList;
import java.util.Date;
import java.util.logging.Logger;

import static mindspore.schema.CompressType.NO_COMPRESS;

import mindspore.schema.CompressType;

/**
 * Define the serialization method, handle the response message returned from server for getModel request.
 *
 * @since 2021-06-30
 */
public class GetModel {
    private static final Logger LOGGER = FLLoggerGenerater.getModelLogger(GetModel.class.toString());
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
            LOGGER.severe("[GetModel] the input parameter of <name> is null or empty, please check!");
            throw new IllegalArgumentException();
        }
        RequestGetModelBuilder builder = new RequestGetModelBuilder();
        return builder.iteration(iteration).flName(name).time()
                .downloadCompressTypesBuilder(flParameter.getDownloadCompressTypes()).build();
    }


    abstract class FeatureGenerator {
        protected ResponseGetModel responseDataBuf;
        protected int curPos = 0;
        protected int size = 0;

        public FeatureGenerator(ResponseGetModel responseDataBuf) {
            this.responseDataBuf = responseDataBuf;
        }

        public abstract FeatureMap next();

        public boolean isEnd() {
            return curPos >= size;
        }
    }

    class NormalFeatureGenerator extends FeatureGenerator {
        public NormalFeatureGenerator(ResponseGetModel responseDataBuf) {
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
        public QuatFeatureGenerator(ResponseGetModel responseDataBuf) {
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

    private FeatureGenerator FeatureGeneratorCtr(ResponseGetModel responseDataBuf) {
        byte compressType = responseDataBuf.downloadCompressType();
        switch (compressType) {
            case CompressType.NO_COMPRESS:
                LOGGER.info("[FeatureGeneratorCtr] Compress type:" + compressType);
                return new NormalFeatureGenerator(responseDataBuf);
            case CompressType.QUANT:
                LOGGER.info("[FeatureGeneratorCtr] Compress type:" + compressType);
                return new QuatFeatureGenerator(responseDataBuf);
            default:
                LOGGER.severe("[FeatureGeneratorCtr] Unsupported compress type:" + compressType);
                return null;
        }
    }

    private FLClientStatus parseResponseFeatures(ResponseGetModel responseDataBuf) {
        Client client = ClientManager.getClient(flParameter.getFlName());
        FeatureGenerator featureGenerator = FeatureGeneratorCtr(responseDataBuf);
        if (localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
            return updateFeatureForHybrid(client, featureGenerator);
        }
        if (localFLParameter.getServerMod().equals(ServerMod.FEDERATED_LEARNING.toString())) {
            return updateFeatureForFederated(client, featureGenerator);
        }
        LOGGER.severe("[parseResponseFeatures] Unsupported ServerMod:" + localFLParameter.getServerMod());
        return FLClientStatus.FAILED;
    }

    private FLClientStatus updateFeatureForFederated(Client client, FeatureGenerator featureGenerator) {
        FLClientStatus result = FLClientStatus.SUCCESS;
        Status status = Status.SUCCESS;
        while (!featureGenerator.isEnd()) {
            FeatureMap featureMap = featureGenerator.next();
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
        while (!featureGenerator.isEnd()) {
            FeatureMap featureMap = featureGenerator.next();
            if (flParameter.getHybridWeightName(RunType.TRAINMODE).contains(featureMap.weightFullname())) {
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

    /**
     * Handle the response message returned from server.
     *
     * @param responseDataBuf the response message returned from server.
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus doResponse(ResponseGetModel responseDataBuf) {
        retCode = responseDataBuf.retcode();
        LOGGER.info("[getModel] ==========the response message of getModel is:================");
        LOGGER.info("[getModel] ==========retCode: " + retCode);
        LOGGER.info("[getModel] ==========reason: " + responseDataBuf.reason());
        LOGGER.info("[getModel] ==========iteration: " + responseDataBuf.iteration());
        LOGGER.info("[getModel] ==========time: " + responseDataBuf.timestamp());
        FLClientStatus status = FLClientStatus.SUCCESS;
        switch (responseDataBuf.retcode()) {
            case (ResponseCode.SUCCEED):
                LOGGER.info("[getModel] into <parseResponseFeatures>");
                status = parseResponseFeatures(responseDataBuf);
                return status;
            case (ResponseCode.SucNotReady):
                LOGGER.info("[getModel] server is not ready now: need wait and request getModel again");
                return FLClientStatus.WAIT;
            case (ResponseCode.OutOfTime):
                LOGGER.info("[getModel] out of time: need wait and request startFLJob again");
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.warning("[getModel] catch RequestError or SystemError");
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe("[getModel] the return <retCode> from server is invalid: " + retCode);
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
                LOGGER.severe("[GetModel] the input parameter of <name> is null or empty, please " +
                        "check!");
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
                LOGGER.severe("[GetModel] the parameter of <downloadCompressTypes> is null or empty," +
                        " please check!");
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
