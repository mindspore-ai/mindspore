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
import com.mindspore.flclient.cipher.struct.EncryptShare;
import com.mindspore.flclient.cipher.struct.NewArray;
import mindspore.schema.GetClientList;
import mindspore.schema.ResponseCode;
import mindspore.schema.ReturnClientList;

import java.nio.ByteBuffer;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import static com.mindspore.flclient.FLParameter.SLEEP_TIME;
import static com.mindspore.flclient.LocalFLParameter.IVEC_LEN;

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

    public FLClientStatus getClientList(int iteration, List<String> u3ClientList, List<DecryptShareSecrets> decryptSecretsList, List<EncryptShare> returnShareList, Map<String, byte[]> cuvKeys) {
        String url = Common.generateUrl(flParameter.isUseHttps(), flParameter.isUseElb(), flParameter.getIp(), flParameter.getPort(), flParameter.getServerNum());
        FlatBufferBuilder builder = new FlatBufferBuilder();
        int id = builder.createString(localFLParameter.getFlID());
        String dateTime = LocalDateTime.now().toString();
        int time = builder.createString(dateTime);
        int clientListRoot = GetClientList.createGetClientList(builder, id, iteration, time);
        builder.finish(clientListRoot);
        byte[] msg = builder.sizedByteArray();
        try {
            byte[] responseData = flCommunication.syncRequest(url + "/getClientList", msg);
            if (Common.isSafeMod(responseData, localFLParameter.getSafeMod())) {
                LOGGER.info(Common.addTag("[getClientList] The cluster is in safemode, need wait some time and request again"));
                Common.sleep(SLEEP_TIME);
                nextRequestTime = "";
                return FLClientStatus.RESTART;
            }
            ByteBuffer buffer = ByteBuffer.wrap(responseData);
            ReturnClientList clientListRsp = ReturnClientList.getRootAsReturnClientList(buffer);
            FLClientStatus status = judgeGetClientList(clientListRsp, u3ClientList, decryptSecretsList, returnShareList, cuvKeys);
            return status;
        } catch (Exception e) {
            e.printStackTrace();
            return FLClientStatus.FAILED;
        }
    }

    public FLClientStatus judgeGetClientList(ReturnClientList bufData, List<String> u3ClientList, List<DecryptShareSecrets> decryptSecretsList, List<EncryptShare> returnShareList, Map<String, byte[]> cuvKeys) {
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
                try {
                    decryptSecretShares(decryptSecretsList, returnShareList, cuvKeys);
                } catch (Exception e) {
                    e.printStackTrace();
                    return FLClientStatus.FAILED;
                }
                return FLClientStatus.SUCCESS;
            case (ResponseCode.SucNotReady):
                LOGGER.info(Common.addTag("[PairWiseMask] server is not ready now, need wait and request GetClientList again!"));
                return FLClientStatus.WAIT;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[PairWiseMask] GetClientList out of time: need wait and request startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch SucNotMatch or SystemError in GetClientList"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ReturnClientList is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    public void decryptSecretShares(List<DecryptShareSecrets> decryptSecretsList, List<EncryptShare> returnShareList, Map<String, byte[]> cuvKeys) throws Exception {
        decryptSecretsList.clear();
        int size = returnShareList.size();
        for (int i = 0; i < size; i++) {
            DecryptShareSecrets decryptShareSecrets = new DecryptShareSecrets();
            EncryptShare encryptShare = returnShareList.get(i);
            String vFlID = encryptShare.getFlID();
            byte[] share = encryptShare.getShare().getArray();
            byte[] iVecIn = new byte[IVEC_LEN];
            AESEncrypt aesEncrypt = new AESEncrypt(cuvKeys.get(vFlID), iVecIn, "CBC");
            byte[] decryptShare = aesEncrypt.decrypt(cuvKeys.get(vFlID), share);
            int sSize = (int) decryptShare[0];
            int bSize = (int) decryptShare[1];
            int sIndexLen = (int) decryptShare[2];
            int bIndexLen = (int) decryptShare[3];
            int sIndex = BaseUtil.byteArray2Integer(Arrays.copyOfRange(decryptShare, 4, 4 + sIndexLen));
            int bIndex = BaseUtil.byteArray2Integer(Arrays.copyOfRange(decryptShare, 4 + sIndexLen, 4 + sIndexLen + bIndexLen));
            byte[] sSkUv = Arrays.copyOfRange(decryptShare, 4 + sIndexLen + bIndexLen, 4 + sIndexLen + bIndexLen + sSize);
            byte[] bUv = Arrays.copyOfRange(decryptShare, 4 + sIndexLen + bIndexLen + sSize, 4 + sIndexLen + bIndexLen + sSize + bSize);
            NewArray<byte[]> sSkVu = new NewArray<>();
            sSkVu.setSize(sSize);
            sSkVu.setArray(sSkUv);
            NewArray bVu = new NewArray();
            bVu.setSize(bSize);
            bVu.setArray(bUv);
            decryptShareSecrets.setFlID(vFlID);
            decryptShareSecrets.setSSkVu(sSkVu);
            decryptShareSecrets.setBVu(bVu);
            decryptShareSecrets.setSIndex(sIndex);
            decryptShareSecrets.setIndexB(bIndex);
            decryptSecretsList.add(decryptShareSecrets);
        }
    }
}
