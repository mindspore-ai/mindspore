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

package com.mindspore.flclient.cipher;

import com.google.flatbuffers.FlatBufferBuilder;
import com.mindspore.flclient.Common;
import com.mindspore.flclient.FLClientStatus;
import com.mindspore.flclient.FLCommunication;
import com.mindspore.flclient.FLParameter;
import com.mindspore.flclient.LocalFLParameter;
import com.mindspore.flclient.cipher.struct.DecryptShareSecrets;
import mindspore.schema.ClientShare;
import mindspore.schema.ResponseCode;

import java.nio.ByteBuffer;
import java.time.LocalDateTime;
import java.util.List;
import java.util.logging.Logger;

public class ReconstructSecretReq {
    private static final Logger LOGGER = Logger.getLogger(ReconstructSecretReq.class.toString());
    private FLCommunication flCommunication;
    private String nextRequestTime;
    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private int retCode;

    public String getNextRequestTime() {
        return nextRequestTime;
    }

    public void setNextRequestTime(String nextRequestTime) {
        this.nextRequestTime = nextRequestTime;
    }

    public int getRetCode() {
        return retCode;
    }

    public ReconstructSecretReq() {
        flCommunication = FLCommunication.getInstance();
    }

    public FLClientStatus sendReconstructSecret(List<DecryptShareSecrets> decryptShareSecretsList, List<String> u3ClientList, int iteration) {
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getIp(), flParameter.getPort(), flParameter.getServerNum());
        LOGGER.info(Common.addTag("[PairWiseMask] ==============sendReconstructSecret url: " + url + "=============="));
        FlatBufferBuilder builder = new FlatBufferBuilder();
        int desFlId = builder.createString(localFLParameter.getFlID());
        String dateTime = LocalDateTime.now().toString();
        int time = builder.createString(dateTime);
        int shareSecretsSize = decryptShareSecretsList.size();
        if (shareSecretsSize <= 0) {
            LOGGER.info(Common.addTag("[PairWiseMask] request failed: the decryptShareSecretsList is null, please waite."));
            return FLClientStatus.FAILED;
        } else {
            int[] decryptShareList = new int[shareSecretsSize];
            for (int i = 0; i < shareSecretsSize; i++) {
                DecryptShareSecrets decryptShareSecrets = decryptShareSecretsList.get(i);
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
            int reconstructShareSecrets = mindspore.schema.SendReconstructSecret.createReconstructSecretSharesVector(builder, decryptShareList);
            int reconstructSecretRoot = mindspore.schema.SendReconstructSecret.createSendReconstructSecret(builder, desFlId, reconstructShareSecrets, iteration, time);
            builder.finish(reconstructSecretRoot);
            byte[] msg = builder.sizedByteArray();
            try {
                byte[] responseData = flCommunication.syncRequest(url + "/reconstructSecrets", msg);
                ByteBuffer buffer = ByteBuffer.wrap(responseData);
                mindspore.schema.ReconstructSecret reconstructSecretRsp = mindspore.schema.ReconstructSecret.getRootAsReconstructSecret(buffer);
                FLClientStatus status = judgeSendReconstructSecrets(reconstructSecretRsp);
                return status;
            } catch (Exception e) {
                LOGGER.severe(Common.addTag("[PairWiseMask] un solved error code in reconstruct"));
                e.printStackTrace();
                return FLClientStatus.FAILED;
            }
        }
    }

    public FLClientStatus judgeSendReconstructSecrets(mindspore.schema.ReconstructSecret bufData) {
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
                LOGGER.info(Common.addTag("[PairWiseMask] SendReconstructSecrets out of time: need wait and request startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch SucNotMatch or SystemError in SendReconstructSecrets"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ReconstructSecret is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }
}
