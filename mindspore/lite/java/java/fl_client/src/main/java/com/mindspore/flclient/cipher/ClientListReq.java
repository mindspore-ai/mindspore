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
import com.mindspore.flclient.cipher.struct.EncryptShare;
import com.mindspore.flclient.cipher.struct.NewArray;

import mindspore.schema.GetClientList;
import mindspore.schema.ResponseCode;
import mindspore.schema.ReturnClientList;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Define the serialization method, handle the response message returned from server for GetClientList request.
 *
 * @since 2021-06-30
 */
public class ClientListReq {
    private static final Logger LOGGER = Logger.getLogger(ClientListReq.class.toString());

    private FLCommunication flCommunication;
    private String nextRequestTime;
    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private int retCode;

    public ClientListReq() {
        flCommunication = FLCommunication.getInstance();
    }

    public String getNextRequestTime() {
        return nextRequestTime;
    }

    public void setNextRequestTime(String nextRequestTime) {
        this.nextRequestTime = nextRequestTime;
    }

    public int getRetCode() {
        return retCode;
    }

    /**
     * Send serialized request message of GetClientList to server.
     *
     * @param iteration          current iteration of federated learning task.
     * @param u3ClientList       list of clients successfully requested in UpdateModel round.
     * @param decryptSecretsList list to store to decrypted secret fragments.
     * @param returnShareList    List of returned secret fragments from server.
     * @param cuvKeys            Keys used to decrypt secret fragments.
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus getClientList(int iteration, List<String> u3ClientList,
                                        List<DecryptShareSecrets> decryptSecretsList,
                                        List<EncryptShare> returnShareList, Map<String, byte[]> cuvKeys) {
        FlatBufferBuilder builder = new FlatBufferBuilder();
        int id = builder.createString(localFLParameter.getFlID());
        Date date = new Date();
        long timestamp = date.getTime();
        String dateTime = String.valueOf(timestamp);
        int time = builder.createString(dateTime);
        int clientListRoot = GetClientList.createGetClientList(builder, id, iteration, time);
        builder.finish(clientListRoot);
        byte[] msg = builder.sizedByteArray();
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(), flParameter.getDomainName());
        try {
            byte[] responseData = flCommunication.syncRequest(url + "/getClientList", msg);
            if (!Common.isSeverReady(responseData)) {
                LOGGER.info(Common.addTag("[getClientList] the server is not ready now, need wait some time and " +
                        "request again"));
                Common.sleep(SLEEP_TIME);
                nextRequestTime = "";
                return FLClientStatus.RESTART;
            }
            ByteBuffer buffer = ByteBuffer.wrap(responseData);
            LOGGER.info(Common.addTag("getClientList responseData size: " + responseData.length));
            ReturnClientList clientListRsp = ReturnClientList.getRootAsReturnClientList(buffer);
            return judgeGetClientList(clientListRsp, u3ClientList, decryptSecretsList, returnShareList, cuvKeys);
        } catch (IOException ex) {
            LOGGER.severe(Common.addTag("[getClientList] unsolved error code in getClientList: catch IOException: " +
                    ex.getMessage()));
            retCode = ResponseCode.RequestError;
            return FLClientStatus.FAILED;
        }
    }

    /**
     * Analyze the serialization message returned from server and perform corresponding processing.
     *
     * @param bufData            Serialized message returned from server.
     * @param u3ClientList       list of clients successfully requested in UpdateModel round.
     * @param decryptSecretsList list to store decrypted secret fragments.
     * @param returnShareList    List of returned secret fragments from server.
     * @param cuvKeys            Keys used to decrypt secret fragments.
     * @return the status code corresponding to the response message.
     */
    private FLClientStatus judgeGetClientList(ReturnClientList bufData, List<String> u3ClientList,
                                              List<DecryptShareSecrets> decryptSecretsList,
                                              List<EncryptShare> returnShareList, Map<String, byte[]> cuvKeys) {
        retCode = bufData.retcode();
        LOGGER.info(Common.addTag("[PairWiseMask] ************** the response of GetClientList **************"));
        LOGGER.info(Common.addTag("[PairWiseMask] return code: " + retCode));
        LOGGER.info(Common.addTag("[PairWiseMask] reason: " + bufData.reason()));
        LOGGER.info(Common.addTag("[PairWiseMask] current iteration in server: " + bufData.iteration()));
        LOGGER.info(Common.addTag("[PairWiseMask] next request time: " + bufData.nextReqTime()));
        LOGGER.info(Common.addTag("[PairWiseMask] the size of clients: " + bufData.clientsLength()));
        FLClientStatus status;
        switch (retCode) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[PairWiseMask] GetClientList success"));
                u3ClientList.clear();
                int clientSize = bufData.clientsLength();
                for (int i = 0; i < clientSize; i++) {
                    String curFlId = bufData.clients(i);
                    u3ClientList.add(curFlId);
                }
                status = decryptSecretShares(decryptSecretsList, returnShareList, cuvKeys);
                return status;
            case (ResponseCode.SucNotReady):
                LOGGER.info(Common.addTag("[PairWiseMask] server is not ready now, need wait and request " +
                        "GetClientList again!"));
                return FLClientStatus.WAIT;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[PairWiseMask] GetClientList out of time: need wait and request startFLJob" +
                        " again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch SucNotMatch or SystemError in GetClientList"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ReturnClientList is " +
                        "invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    private FLClientStatus decryptSecretShares(List<DecryptShareSecrets> decryptSecretsList,
                                               List<EncryptShare> returnShareList, Map<String, byte[]> cuvKeys) {
        decryptSecretsList.clear();
        int size = returnShareList.size();
        if (size <= 0) {
            LOGGER.severe(Common.addTag("[PairWiseMask] the input argument <returnShareList> is null"));
            return FLClientStatus.FAILED;
        }
        if (cuvKeys.isEmpty()) {
            LOGGER.severe(Common.addTag("[PairWiseMask] the input argument <cuvKeys> is null"));
            return FLClientStatus.FAILED;
        }
        for (int i = 0; i < size; i++) {
            EncryptShare encryptShare = returnShareList.get(i);
            String vFlID = encryptShare.getFlID();
            byte[] share = encryptShare.getShare().getArray();
            if (!cuvKeys.containsKey(vFlID)) {
                LOGGER.severe(Common.addTag("[PairWiseMask] the key <vFlID> is not in map <cuvKeys> "));
                return FLClientStatus.FAILED;
            }
            AESEncrypt aesEncrypt = new AESEncrypt(cuvKeys.get(vFlID), "CBC");
            byte[] decryptShare = aesEncrypt.decrypt(cuvKeys.get(vFlID), share);
            if (decryptShare == null || decryptShare.length == 0) {
                LOGGER.severe(Common.addTag("[decryptSecretShares] the return byte[] is null, please check!"));
                return FLClientStatus.FAILED;
            }
            if (decryptShare.length < 4) {
                LOGGER.severe(Common.addTag("[decryptSecretShares] the returned decryptShare is not valid: length is " +
                        "not right, please check!"));
                return FLClientStatus.FAILED;
            }
            int sSize = (int) decryptShare[0];
            int bSize = (int) decryptShare[1];
            int sIndexLen = (int) decryptShare[2];
            int bIndexLen = (int) decryptShare[3];
            if (decryptShare.length < (4 + sIndexLen + bIndexLen + sSize + bSize)) {
                LOGGER.severe(Common.addTag("[decryptSecretShares] the returned decryptShare is not valid: length is " +
                        "not right, please check!"));
                return FLClientStatus.FAILED;
            }
            byte[] sSkUv = Arrays.copyOfRange(decryptShare, 4 + sIndexLen + bIndexLen,
                    4 + sIndexLen + bIndexLen + sSize);
            byte[] bUv = Arrays.copyOfRange(decryptShare, 4 + sIndexLen + bIndexLen + sSize,
                    4 + sIndexLen + bIndexLen + sSize + bSize);
            NewArray<byte[]> sSkVu = new NewArray<>();
            sSkVu.setSize(sSize);
            sSkVu.setArray(sSkUv);
            NewArray bVu = new NewArray();
            bVu.setSize(bSize);
            bVu.setArray(bUv);
            int sIndex = BaseUtil.byteArray2Integer(Arrays.copyOfRange(decryptShare, 4, 4 + sIndexLen));
            int bIndex = BaseUtil.byteArray2Integer(Arrays.copyOfRange(decryptShare, 4 + sIndexLen,
                    4 + sIndexLen + bIndexLen));
            DecryptShareSecrets decryptShareSecrets = new DecryptShareSecrets();
            decryptShareSecrets.setFlID(vFlID);
            decryptShareSecrets.setSSkVu(sSkVu);
            decryptShareSecrets.setBVu(bVu);
            decryptShareSecrets.setSIndex(sIndex);
            decryptShareSecrets.setIndexB(bIndex);
            decryptSecretsList.add(decryptShareSecrets);
        }
        return FLClientStatus.SUCCESS;
    }
}
