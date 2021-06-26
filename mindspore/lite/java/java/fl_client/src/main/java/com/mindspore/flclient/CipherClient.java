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
import com.mindspore.flclient.cipher.AESEncrypt;
import com.mindspore.flclient.cipher.BaseUtil;
import com.mindspore.flclient.cipher.ClientListReq;
import com.mindspore.flclient.cipher.KEYAgreement;
import com.mindspore.flclient.cipher.Random;
import com.mindspore.flclient.cipher.ReconstructSecretReq;
import com.mindspore.flclient.cipher.ShareSecrets;
import com.mindspore.flclient.cipher.struct.ClientPublicKey;
import com.mindspore.flclient.cipher.struct.DecryptShareSecrets;
import com.mindspore.flclient.cipher.struct.EncryptShare;
import com.mindspore.flclient.cipher.struct.NewArray;
import com.mindspore.flclient.cipher.struct.ShareSecret;
import mindspore.schema.ClientShare;
import mindspore.schema.GetExchangeKeys;
import mindspore.schema.GetShareSecrets;
import mindspore.schema.RequestExchangeKeys;
import mindspore.schema.RequestShareSecrets;
import mindspore.schema.ResponseCode;
import mindspore.schema.ResponseExchangeKeys;
import mindspore.schema.ResponseShareSecrets;
import mindspore.schema.ReturnExchangeKeys;
import mindspore.schema.ReturnShareSecrets;

import java.io.UnsupportedEncodingException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.security.spec.InvalidKeySpecException;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import static com.mindspore.flclient.FLParameter.SLEEP_TIME;
import static com.mindspore.flclient.LocalFLParameter.IVEC_LEN;
import static com.mindspore.flclient.LocalFLParameter.SEED_SIZE;

public class CipherClient {
    private static final Logger LOGGER = Logger.getLogger(CipherClient.class.toString());
    private FLCommunication flCommunication;
    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private final int iteration;
    private int featureSize;
    private int t;
    private List<byte[]> cKey = new ArrayList<>();
    private List<byte[]> sKey = new ArrayList<>();
    private byte[] bu;
    private String nextRequestTime;
    private Map<String, ClientPublicKey> clientPublicKeyList = new HashMap<String, ClientPublicKey>();
    private Map<String, byte[]> sUVKeys = new HashMap<String, byte[]>();
    private Map<String, byte[]> cUVKeys = new HashMap<String, byte[]>();
    private List<EncryptShare> clientShareList = new ArrayList<>();
    private List<EncryptShare> returnShareList = new ArrayList<>();
    private float[] featureMask;
    private List<String> u1ClientList = new ArrayList<>();
    private List<String> u2UClientList = new ArrayList<>();
    private List<String> u3ClientList = new ArrayList<>();
    private List<DecryptShareSecrets> decryptShareSecretsList = new ArrayList<>();
    private byte[] prime;
    private KEYAgreement keyAgreement = new KEYAgreement();
    private Random random = new Random();
    private ClientListReq clientListReq = new ClientListReq();
    private ReconstructSecretReq reconstructSecretReq = new ReconstructSecretReq();
    private int retCode;

    public CipherClient(int iter, int minSecretNum, byte[] prime, int featureSize) {
        flCommunication = FLCommunication.getInstance();
        this.iteration = iter;
        this.featureSize = featureSize;
        this.t = minSecretNum;
        this.prime = prime;
        this.featureMask = new float[this.featureSize];
    }

    public void setNextRequestTime(String nextRequestTime) {
        this.nextRequestTime = nextRequestTime;
    }

    public void setBU(byte[] bu) {
        this.bu = bu;
    }

    public void setClientShareList(List<EncryptShare> clientShareList) {
        this.clientShareList.clear();
        this.clientShareList = clientShareList;
    }

    public String getNextRequestTime() {
        return nextRequestTime;
    }

    public int getRetCode() {
        return retCode;
    }

    public void genDHKeyPairs() {
        byte[] csk = keyAgreement.generatePrivateKey();
        byte[] cpk = keyAgreement.generatePublicKey(csk);
        byte[] ssk = keyAgreement.generatePrivateKey();
        byte[] spk = keyAgreement.generatePublicKey(ssk);
        this.cKey.add(cpk);
        this.cKey.add(csk);
        this.sKey.add(spk);
        this.sKey.add(ssk);
    }

    public void genIndividualSecret() {
        byte[] key = new byte[SEED_SIZE];
        random.getRandomBytes(key);
        setBU(key);
    }

    public List<ShareSecret> genSecretShares(byte[] secret) throws UnsupportedEncodingException {
        List<ShareSecret> shareSecretList = new ArrayList<>();
        int size = u1ClientList.size();
        ShareSecrets shamir = new ShareSecrets(t, size - 1);
        ShareSecrets.SecretShare[] shares = shamir.split(secret, prime);
        int j = 0;
        for (int i = 0; i < size; i++) {
            String vFlID = u1ClientList.get(i);
            if (localFLParameter.getFlID().equals(vFlID)) {
                continue;
            } else {
                ShareSecret shareSecret = new ShareSecret();
                NewArray<byte[]> array = new NewArray<>();
                int index = shares[j].getNum();
                BigInteger intShare = shares[j].getShare();
                byte[] share = BaseUtil.bigInteger2byteArray(intShare);
                array.setSize(share.length);
                array.setArray(share);
                shareSecret.setFlID(vFlID);
                shareSecret.setShare(array);
                shareSecret.setIndex(index);
                shareSecretList.add(shareSecret);
                j += 1;
            }
        }
        return shareSecretList;
    }

    public void genEncryptExchangedKeys() throws InvalidKeySpecException, NoSuchAlgorithmException {
        cUVKeys.clear();
        for (String key : clientPublicKeyList.keySet()) {
            ClientPublicKey curPublicKey = clientPublicKeyList.get(key);
            String vFlID = curPublicKey.getFlID();
            if (localFLParameter.getFlID().equals(vFlID)) {
                continue;
            } else {
                byte[] secret1 = keyAgreement.keyAgreement(cKey.get(1), curPublicKey.getCPK().getArray());
                byte[] salt = new byte[0];
                byte[] secret = keyAgreement.getEncryptedPassword(secret1, salt);
                cUVKeys.put(vFlID, secret);
            }
        }
    }

    public void encryptShares() throws Exception {
        LOGGER.info(Common.addTag("[PairWiseMask] ************** generate encrypt share secrets for RequestShareSecrets **************"));
        List<EncryptShare> encryptShareList = new ArrayList<>();
        // connect sSkUv, bUV, sIndex, indexB  and  then Encrypt them
        List<ShareSecret> sSkUv = genSecretShares(sKey.get(1));
        List<ShareSecret> bUV = genSecretShares(bu);
        for (int i = 0; i < bUV.size(); i++) {
            EncryptShare encryptShare = new EncryptShare();
            NewArray<byte[]> array = new NewArray<>();
            String vFlID = bUV.get(i).getFlID();
            byte[] sShare = sSkUv.get(i).getShare().getArray();
            byte[] bShare = bUV.get(i).getShare().getArray();
            byte[] sIndex = BaseUtil.integer2byteArray(sSkUv.get(i).getIndex());
            byte[] bIndex = BaseUtil.integer2byteArray(bUV.get(i).getIndex());
            byte[] allSecret = new byte[sShare.length + bShare.length + sIndex.length + bIndex.length + 4];
            allSecret[0] = (byte) sShare.length;
            allSecret[1] = (byte) bShare.length;
            allSecret[2] = (byte) sIndex.length;
            allSecret[3] = (byte) bIndex.length;
            System.arraycopy(sIndex, 0, allSecret, 4, sIndex.length);
            System.arraycopy(bIndex, 0, allSecret, 4 + sIndex.length, bIndex.length);
            System.arraycopy(sShare, 0, allSecret, 4 + sIndex.length + bIndex.length, sShare.length);
            System.arraycopy(bShare, 0, allSecret, 4 + sIndex.length + bIndex.length + sShare.length, bShare.length);
            // encrypt:
            byte[] iVecIn = new byte[IVEC_LEN];
            AESEncrypt aesEncrypt = new AESEncrypt(cUVKeys.get(vFlID), iVecIn, "CBC");
            byte[] encryptData = aesEncrypt.encrypt(cUVKeys.get(vFlID), allSecret);
            array.setSize(encryptData.length);
            array.setArray(encryptData);
            encryptShare.setFlID(vFlID);
            encryptShare.setShare(array);
            encryptShareList.add(encryptShare);
        }
        setClientShareList(encryptShareList);
    }

    public float[] doubleMaskingWeight() throws Exception {
        int size = u2UClientList.size();
        List<Float> noiseBu = new ArrayList<>();
        random.randomAESCTR(noiseBu, featureSize, bu);
        float[] mask = new float[featureSize];
        for (int i = 0; i < size; i++) {
            String vFlID = u2UClientList.get(i);
            ClientPublicKey curPublicKey = clientPublicKeyList.get(vFlID);
            if (localFLParameter.getFlID().equals(vFlID)) {
                continue;
            } else {
                byte[] salt = new byte[0];
                byte[] secret1 = keyAgreement.keyAgreement(sKey.get(1), curPublicKey.getSPK().getArray());
                byte[] secret = keyAgreement.getEncryptedPassword(secret1, salt);
                sUVKeys.put(vFlID, secret);
                List<Float> noiseSuv = new ArrayList<>();
                random.randomAESCTR(noiseSuv, featureSize, secret);
                int sign;
                if (localFLParameter.getFlID().compareTo(vFlID) > 0) {
                    sign = 1;
                } else {
                    sign = -1;
                }
                for (int j = 0; j < noiseSuv.size(); j++) {
                    mask[j] = mask[j] + sign * noiseSuv.get(j);
                }
            }
        }
        for (int j = 0; j < noiseBu.size(); j++) {
            mask[j] = mask[j] + noiseBu.get(j);
        }
        return mask;
    }

    public NewArray<byte[]> byteToArray(ByteBuffer buf, int size) {
        NewArray<byte[]> newArray = new NewArray<>();
        newArray.setSize(size);
        byte[] array = new byte[size];
        for (int i = 0; i < size; i++) {
            byte word = buf.get();
            array[i] = word;
        }
        newArray.setArray(array);
        return newArray;
    }

    public FLClientStatus requestExchangeKeys() {
        LOGGER.info(Common.addTag("[PairWiseMask] ==============request flID: " + localFLParameter.getFlID() + "=============="));
        String url = Common.generateUrl(flParameter.isUseHttps(), flParameter.isUseElb(), flParameter.getIp(), flParameter.getPort(), flParameter.getServerNum());
        LOGGER.info(Common.addTag("[PairWiseMask] ==============requestExchangeKeys url: " + url + "=============="));
        genDHKeyPairs();
        byte[] cPK = cKey.get(0);
        byte[] sPK = sKey.get(0);
        FlatBufferBuilder fbBuilder = new FlatBufferBuilder();
        int id = fbBuilder.createString(localFLParameter.getFlID());
        int cpk = RequestExchangeKeys.createCPkVector(fbBuilder, cPK);
        int spk = RequestExchangeKeys.createSPkVector(fbBuilder, sPK);
        String dateTime = LocalDateTime.now().toString();
        int time = fbBuilder.createString(dateTime);
        int exchangeKeysRoot = RequestExchangeKeys.createRequestExchangeKeys(fbBuilder, id, cpk, spk, iteration, time);
        fbBuilder.finish(exchangeKeysRoot);
        byte[] msg = fbBuilder.sizedByteArray();
        try {
            byte[] responseData = flCommunication.syncRequest(url + "/exchangeKeys", msg);
            ByteBuffer buffer = ByteBuffer.wrap(responseData);
            ResponseExchangeKeys responseExchangeKeys = ResponseExchangeKeys.getRootAsResponseExchangeKeys(buffer);
            FLClientStatus status = judgeRequestExchangeKeys(responseExchangeKeys);
            return status;
        } catch (Exception e) {
            e.printStackTrace();
            return FLClientStatus.FAILED;
        }
    }

    public FLClientStatus judgeRequestExchangeKeys(ResponseExchangeKeys bufData) {
        retCode = bufData.retcode();
        LOGGER.info(Common.addTag("[PairWiseMask] **************the response of RequestExchangeKeys**************"));
        LOGGER.info(Common.addTag("[PairWiseMask] return code: " + retCode));
        LOGGER.info(Common.addTag("[PairWiseMask] reason: " + bufData.reason()));
        LOGGER.info(Common.addTag("[PairWiseMask] current iteration in server: " + bufData.iteration()));
        LOGGER.info(Common.addTag("[PairWiseMask] next request time: " + bufData.nextReqTime()));
        switch (retCode) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[PairWiseMask] RequestExchangeKeys success"));
                return FLClientStatus.SUCCESS;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[PairWiseMask] RequestExchangeKeys out of time: need wait and request startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch RequestError or SystemError in RequestExchangeKeys"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ResponseExchangeKeys is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    public FLClientStatus getExchangeKeys() {
        String url = Common.generateUrl(flParameter.isUseHttps(), flParameter.isUseElb(), flParameter.getIp(), flParameter.getPort(), flParameter.getServerNum());
        LOGGER.info(Common.addTag("[PairWiseMask] ==============getExchangeKeys url: " + url + "=============="));
        FlatBufferBuilder fbBuilder = new FlatBufferBuilder();
        int id = fbBuilder.createString(localFLParameter.getFlID());
        String dateTime = LocalDateTime.now().toString();
        int time = fbBuilder.createString(dateTime);
        int getExchangeKeysRoot = GetExchangeKeys.createGetExchangeKeys(fbBuilder, id, iteration, time);
        fbBuilder.finish(getExchangeKeysRoot);
        byte[] msg = fbBuilder.sizedByteArray();
        try {
            byte[] responseData = flCommunication.syncRequest(url + "/getKeys", msg);
            ByteBuffer buffer = ByteBuffer.wrap(responseData);
            ReturnExchangeKeys returnExchangeKeys = ReturnExchangeKeys.getRootAsReturnExchangeKeys(buffer);
            FLClientStatus status = judgeGetExchangeKeys(returnExchangeKeys);
            return status;
        } catch (Exception e) {
            e.printStackTrace();
            return FLClientStatus.FAILED;
        }
    }

    public FLClientStatus judgeGetExchangeKeys(ReturnExchangeKeys bufData) {
        retCode = bufData.retcode();
        LOGGER.info(Common.addTag("[PairWiseMask] **************the response of GetExchangeKeys**************"));
        LOGGER.info(Common.addTag("[PairWiseMask] return code: " + retCode));
        LOGGER.info(Common.addTag("[PairWiseMask] current iteration in server: " + bufData.iteration()));
        LOGGER.info(Common.addTag("[PairWiseMask] next request time: " + bufData.nextReqTime()));
        switch (retCode) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[PairWiseMask] GetExchangeKeys success"));
                clientPublicKeyList.clear();
                u1ClientList.clear();
                int length = bufData.remotePublickeysLength();
                for (int i = 0; i < length; i++) {
                    ClientPublicKey publicKey = new ClientPublicKey();
                    String srcFlId = bufData.remotePublickeys(i).flId();
                    publicKey.setFlID(srcFlId);
                    ByteBuffer bufCpk = bufData.remotePublickeys(i).cPkAsByteBuffer();
                    int sizeCpk = bufData.remotePublickeys(i).cPkLength();
                    ByteBuffer bufSpk = bufData.remotePublickeys(i).sPkAsByteBuffer();
                    int sizeSpk = bufData.remotePublickeys(i).sPkLength();
                    publicKey.setCPK(byteToArray(bufCpk, sizeCpk));
                    publicKey.setSPK(byteToArray(bufSpk, sizeSpk));
                    clientPublicKeyList.put(srcFlId, publicKey);
                    u1ClientList.add(srcFlId);
                }
                return FLClientStatus.SUCCESS;
            case (ResponseCode.SucNotReady):
                LOGGER.info(Common.addTag("[PairWiseMask] server is not ready now, need wait and request GetExchangeKeys again!"));
                return FLClientStatus.WAIT;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[PairWiseMask] GetExchangeKeys out of time: need wait and request startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch SucNotMatch or SystemError in GetExchangeKeys"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ReturnExchangeKeys is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    public FLClientStatus requestShareSecrets() throws Exception {
        String url = Common.generateUrl(flParameter.isUseHttps(), flParameter.isUseElb(), flParameter.getIp(), flParameter.getPort(), flParameter.getServerNum());
        LOGGER.info(Common.addTag("[PairWiseMask] ==============requestShareSecrets url: " + url + "=============="));
        genIndividualSecret();
        genEncryptExchangedKeys();
        encryptShares();

        FlatBufferBuilder fbBuilder = new FlatBufferBuilder();
        int id = fbBuilder.createString(localFLParameter.getFlID());
        String dateTime = LocalDateTime.now().toString();
        int time = fbBuilder.createString(dateTime);
        int clientShareSize = clientShareList.size();
        if (clientShareSize <= 0) {
            LOGGER.warning(Common.addTag("[PairWiseMask] encrypt shares is not ready now!"));
            Common.sleep(SLEEP_TIME);
            FLClientStatus status = requestShareSecrets();
            return status;
        } else {
            int[] add = new int[clientShareSize];
            for (int i = 0; i < clientShareSize; i++) {
                int flID = fbBuilder.createString(clientShareList.get(i).getFlID());
                int shareSecretFbs = ClientShare.createShareVector(fbBuilder, clientShareList.get(i).getShare().getArray());
                ClientShare.startClientShare(fbBuilder);
                ClientShare.addFlId(fbBuilder, flID);
                ClientShare.addShare(fbBuilder, shareSecretFbs);
                int clientShareRoot = ClientShare.endClientShare(fbBuilder);
                add[i] = clientShareRoot;
            }
            int encryptedSharesFbs = RequestShareSecrets.createEncryptedSharesVector(fbBuilder, add);
            int requestShareSecretsRoot = RequestShareSecrets.createRequestShareSecrets(fbBuilder, id, encryptedSharesFbs, iteration, time);
            fbBuilder.finish(requestShareSecretsRoot);
            byte[] msg = fbBuilder.sizedByteArray();
            try {
                byte[] responseData = flCommunication.syncRequest(url + "/shareSecrets", msg);
                ByteBuffer buffer = ByteBuffer.wrap(responseData);
                ResponseShareSecrets responseShareSecrets = ResponseShareSecrets.getRootAsResponseShareSecrets(buffer);
                FLClientStatus status = judgeRequestShareSecrets(responseShareSecrets);
                return status;
            } catch (Exception e) {
                e.printStackTrace();
                return FLClientStatus.FAILED;
            }
        }
    }

    public FLClientStatus judgeRequestShareSecrets(ResponseShareSecrets bufData) {
        retCode = bufData.retcode();
        LOGGER.info(Common.addTag("[PairWiseMask] **************the response of RequestShareSecrets**************"));
        LOGGER.info(Common.addTag("[PairWiseMask] return code: " + retCode));
        LOGGER.info(Common.addTag("[PairWiseMask] reason: " + bufData.reason()));
        LOGGER.info(Common.addTag("[PairWiseMask] current iteration in server: " + bufData.iteration()));
        LOGGER.info(Common.addTag("[PairWiseMask] next request time: " + bufData.nextReqTime()));
        switch (retCode) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[PairWiseMask] RequestShareSecrets success"));
                return FLClientStatus.SUCCESS;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[PairWiseMask] RequestShareSecrets out of time: need wait and request startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch SucNotMatch or SystemError in RequestShareSecrets"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ResponseShareSecrets is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    public FLClientStatus getShareSecrets() {
        String url = Common.generateUrl(flParameter.isUseHttps(), flParameter.isUseElb(), flParameter.getIp(), flParameter.getPort(), flParameter.getServerNum());
        LOGGER.info(Common.addTag("[PairWiseMask] ==============getShareSecrets url: " + url + "=============="));
        FlatBufferBuilder fbBuilder = new FlatBufferBuilder();
        int id = fbBuilder.createString(localFLParameter.getFlID());
        String dateTime = LocalDateTime.now().toString();
        int time = fbBuilder.createString(dateTime);
        int getShareSecrets = GetShareSecrets.createGetShareSecrets(fbBuilder, id, iteration, time);
        fbBuilder.finish(getShareSecrets);
        byte[] msg = fbBuilder.sizedByteArray();
        try {
            byte[] responseData = flCommunication.syncRequest(url + "/getSecrets", msg);
            ByteBuffer buffer = ByteBuffer.wrap(responseData);
            ReturnShareSecrets returnShareSecrets = ReturnShareSecrets.getRootAsReturnShareSecrets(buffer);
            FLClientStatus status = judgeGetShareSecrets(returnShareSecrets);
            return status;
        } catch (Exception e) {
            e.printStackTrace();
            return FLClientStatus.FAILED;
        }
    }

    public FLClientStatus judgeGetShareSecrets(ReturnShareSecrets bufData) {
        retCode = bufData.retcode();
        LOGGER.info(Common.addTag("[PairWiseMask] **************the response of GetShareSecrets**************"));
        LOGGER.info(Common.addTag("[PairWiseMask] return code: " + retCode));
        LOGGER.info(Common.addTag("[PairWiseMask] current iteration in server: " + bufData.iteration()));
        LOGGER.info(Common.addTag("[PairWiseMask] next request time: " + bufData.nextReqTime()));
        LOGGER.info(Common.addTag("[PairWiseMask] the size of encrypted shares: " + bufData.encryptedSharesLength()));
        switch (retCode) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[PairWiseMask] GetShareSecrets success"));
                returnShareList.clear();
                u2UClientList.clear();
                int length = bufData.encryptedSharesLength();
                for (int i = 0; i < length; i++) {
                    EncryptShare shareSecret = new EncryptShare();
                    shareSecret.setFlID(bufData.encryptedShares(i).flId());
                    ByteBuffer bufShare = bufData.encryptedShares(i).shareAsByteBuffer();
                    int sizeShare = bufData.encryptedShares(i).shareLength();
                    shareSecret.setShare(byteToArray(bufShare, sizeShare));
                    returnShareList.add(shareSecret);
                    u2UClientList.add(bufData.encryptedShares(i).flId());
                }

                return FLClientStatus.SUCCESS;
            case (ResponseCode.SucNotReady):
                LOGGER.info(Common.addTag("[PairWiseMask] server is not ready now, need wait and request GetShareSecrets again!"));
                return FLClientStatus.WAIT;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[PairWiseMask] GetShareSecrets out of time: need wait and request startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch SucNotMatch or SystemError in GetShareSecrets"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ReturnShareSecrets is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    public FLClientStatus exchangeKeys() {
        LOGGER.info(Common.addTag("[PairWiseMask] ==================== round0: RequestExchangeKeys+GetExchangeKeys ======================"));
        FLClientStatus curStatus;
        // RequestExchangeKeys
        curStatus = requestExchangeKeys();
        while (curStatus == FLClientStatus.WAIT) {
            Common.sleep(SLEEP_TIME);
            curStatus = requestExchangeKeys();
        }
        if (curStatus != FLClientStatus.SUCCESS) {
            return curStatus;
        }

        // GetExchangeKeys
        curStatus = getExchangeKeys();
        while (curStatus == FLClientStatus.WAIT) {
            Common.sleep(SLEEP_TIME);
            curStatus = getExchangeKeys();
        }
        return curStatus;
    }

    public FLClientStatus shareSecrets() throws Exception {
        LOGGER.info(Common.addTag(("[PairWiseMask] ==================== round1: RequestShareSecrets+GetShareSecrets ======================")));
        FLClientStatus curStatus;
        // RequestShareSecrets
        curStatus = requestShareSecrets();
        while (curStatus == FLClientStatus.WAIT) {
            Common.sleep(SLEEP_TIME);
            curStatus = requestShareSecrets();
        }
        if (curStatus != FLClientStatus.SUCCESS) {
            return curStatus;
        }

        // GetShareSecrets
        curStatus = getShareSecrets();
        while (curStatus == FLClientStatus.WAIT) {
            Common.sleep(SLEEP_TIME);
            curStatus = getShareSecrets();
        }
        return curStatus;
    }

    public FLClientStatus reconstructSecrets() {
        LOGGER.info(Common.addTag("[PairWiseMask] =================== round3: GetClientList+SendReconstructSecret ========================"));
        FLClientStatus curStatus;
        // GetClientList
        curStatus = clientListReq.getClientList(iteration, u3ClientList, decryptShareSecretsList, returnShareList, cUVKeys);
        while (curStatus == FLClientStatus.WAIT) {
            Common.sleep(SLEEP_TIME);
            curStatus = clientListReq.getClientList(iteration, u3ClientList, decryptShareSecretsList, returnShareList, cUVKeys);
        }
        if (curStatus == FLClientStatus.RESTART) {
            nextRequestTime = clientListReq.getNextRequestTime();
        }
        if (curStatus != FLClientStatus.SUCCESS) {
            return curStatus;
        }
        retCode = clientListReq.getRetCode();

        // SendReconstructSecret
        curStatus = reconstructSecretReq.sendReconstructSecret(decryptShareSecretsList, u3ClientList, iteration);
        while (curStatus == FLClientStatus.WAIT) {
            Common.sleep(SLEEP_TIME);
            curStatus = reconstructSecretReq.sendReconstructSecret(decryptShareSecretsList, u3ClientList, iteration);
        }
        if (curStatus == FLClientStatus.RESTART) {
            nextRequestTime = reconstructSecretReq.getNextRequestTime();
        }
        retCode = reconstructSecretReq.getRetCode();
        return curStatus;
    }
}