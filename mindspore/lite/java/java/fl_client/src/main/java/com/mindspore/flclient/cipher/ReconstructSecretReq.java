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

package com.mindspore.flclient.cipher;

import static com.mindspore.flclient.FLParameter.SLEEP_TIME;

import com.google.flatbuffers.FlatBufferBuilder;

import com.mindspore.flclient.Common;
import com.mindspore.flclient.FLClientStatus;
import com.mindspore.flclient.FLCommunication;
import com.mindspore.flclient.FLParameter;
import com.mindspore.flclient.LocalFLParameter;
import com.mindspore.flclient.cipher.struct.DecryptShareSecrets;

import mindspore.schema.ClientShare;
import mindspore.schema.ReconstructSecret;
import mindspore.schema.ResponseCode;
import mindspore.schema.SendReconstructSecret;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Date;
import java.util.List;
import java.util.logging.Logger;

/**
 * reconstruct secret request
 *
 * @since 2021-8-27
 */
public class ReconstructSecretReq {
    private static final Logger LOGGER = Logger.getLogger(ReconstructSecretReq.class.toString());
    private FLCommunication flCommunication;
    private String nextRequestTime;
    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private int retCode;

    /**
     * reconstruct secret request
     */
    public ReconstructSecretReq() {
        flCommunication = FLCommunication.getInstance();
    }

    /**
     * send secret shards to server
     *
     * @param decryptShareSecretsList secret shards list
     * @param u3ClientList            u3 client list
     * @param iteration               iter number
     * @return request result
     */
    public FLClientStatus sendReconstructSecret(List<DecryptShareSecrets> decryptShareSecretsList,
                                                List<String> u3ClientList, int iteration) {
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(), flParameter.getDomainName());
        FlatBufferBuilder builder = new FlatBufferBuilder();
        int desFlId = builder.createString(localFLParameter.getFlID());
        Date date = new Date();
        long timestamp = date.getTime();
        String dateTime = String.valueOf(timestamp);
        int time = builder.createString(dateTime);
        int shareSecretsSize = decryptShareSecretsList.size();
        if (shareSecretsSize <= 0) {
            LOGGER.info(Common.addTag("[PairWiseMask] request failed: the decryptShareSecretsList is null, please " +
                    "waite."));
            return FLClientStatus.FAILED;
        } else {
            int[] decryptShareList = new int[shareSecretsSize];
            for (int i = 0; i < shareSecretsSize; i++) {
                DecryptShareSecrets decryptShareSecrets = decryptShareSecretsList.get(i);
                if (decryptShareSecrets.getFlID() == null) {
                    LOGGER.severe(Common.addTag("[PairWiseMask] get remote flID failed!"));
                    return FLClientStatus.FAILED;
                }

                String srcFlId = decryptShareSecrets.getFlID();
                byte[] share;
                int index;
                if (u3ClientList.contains(srcFlId)) {
                    share = decryptShareSecrets.getBVu().getArray();
                    index = decryptShareSecrets.getIndexB();
                } else {
                    share = decryptShareSecrets.getSSkVu().getArray();
                    index = decryptShareSecrets.getSIndex();
                }
                int fbsSrcFlId = builder.createString(srcFlId);
                int fbsShare = ClientShare.createShareVector(builder, share);
                int clientShare = ClientShare.createClientShare(builder, fbsSrcFlId, fbsShare, index);
                decryptShareList[i] = clientShare;
            }
            int reconstructShareSecrets = SendReconstructSecret.createReconstructSecretSharesVector(builder,
                    decryptShareList);
            int reconstructSecretRoot = SendReconstructSecret.createSendReconstructSecret(builder, desFlId,
                    reconstructShareSecrets, iteration, time);
            builder.finish(reconstructSecretRoot);
            byte[] msg = builder.sizedByteArray();
            try {
                byte[] responseData = flCommunication.syncRequest(url + "/reconstructSecrets", msg);
                if (!Common.isSeverReady(responseData)) {
                    LOGGER.info(Common.addTag("[sendReconstructSecret] the server is not ready now, need wait some " +
                            "time and request again"));
                    Common.sleep(SLEEP_TIME);
                    nextRequestTime = "";
                    return FLClientStatus.RESTART;
                }
                ByteBuffer buffer = ByteBuffer.wrap(responseData);
                ReconstructSecret reconstructSecretRsp = ReconstructSecret.getRootAsReconstructSecret(buffer);
                return judgeSendReconstructSecrets(reconstructSecretRsp);
            } catch (IOException ex) {
                LOGGER.severe(Common.addTag("[PairWiseMask] un solved error code in reconstruct"));
                ex.printStackTrace();
                return FLClientStatus.FAILED;
            }
        }
    }

    private FLClientStatus judgeSendReconstructSecrets(ReconstructSecret bufData) {
        retCode = bufData.retcode();
        LOGGER.info(Common.addTag("[PairWiseMask] **************the response of SendReconstructSecrets**************"));
        LOGGER.info(Common.addTag("[PairWiseMask] return code: " + retCode));
        LOGGER.info(Common.addTag("[PairWiseMask] reason: " + bufData.reason()));
        LOGGER.info(Common.addTag("[PairWiseMask] current iteration in server: " + bufData.iteration()));
        LOGGER.info(Common.addTag("[PairWiseMask] next request time: " + bufData.nextReqTime()));
        switch (retCode) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[PairWiseMask] ReconstructSecrets success"));
                return FLClientStatus.SUCCESS;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[PairWiseMask] SendReconstructSecrets out of time: need wait and request " +
                        "startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch SucNotMatch or SystemError in SendReconstructSecrets"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ReconstructSecret is " +
                        "invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    /**
     * get next request time
     *
     * @return next request time
     */
    public String getNextRequestTime() {
        return nextRequestTime;
    }

    /**
     * set next request time
     *
     * @param nextRequestTime next request time
     */
    public void setNextRequestTime(String nextRequestTime) {
        this.nextRequestTime = nextRequestTime;
    }

    /**
     * get retCode
     *
     * @return retCode
     */
    public int getRetCode() {
        return retCode;
    }
}
