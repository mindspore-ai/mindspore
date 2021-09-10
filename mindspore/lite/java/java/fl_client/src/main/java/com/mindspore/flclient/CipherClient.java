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

import static com.mindspore.flclient.FLParameter.SLEEP_TIME;
import static com.mindspore.flclient.LocalFLParameter.I_VEC_LEN;
import static com.mindspore.flclient.LocalFLParameter.SALT_SIZE;
import static com.mindspore.flclient.LocalFLParameter.SEED_SIZE;

import com.google.flatbuffers.FlatBufferBuilder;

import com.mindspore.flclient.cipher.AESEncrypt;
import com.mindspore.flclient.cipher.BaseUtil;
import com.mindspore.flclient.cipher.ClientListReq;
import com.mindspore.flclient.cipher.KEYAgreement;
import com.mindspore.flclient.cipher.Masking;
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

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * A class used for secure aggregation
 *
 * @since 2021-8-27
 */
public class CipherClient {
    private static final Logger LOGGER = Logger.getLogger(CipherClient.class.toString());
    private FLCommunication flCommunication;
    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private final int iteration;
    private int featureSize;
    private int minShareNum;
    private List<byte[]> cKey = new ArrayList<>();
    private List<byte[]> sKey = new ArrayList<>();
    private byte[] bu;
    private byte[] individualIv = new byte[I_VEC_LEN];
    private byte[] pwIVec = new byte[I_VEC_LEN];
    private byte[] pwSalt = new byte[SALT_SIZE];
    private String nextRequestTime;
    private Map<String, ClientPublicKey> clientPublicKeyList = new HashMap<String, ClientPublicKey>();
    private Map<String, byte[]> sUVKeys = new HashMap<String, byte[]>();
    private Map<String, byte[]> cUVKeys = new HashMap<String, byte[]>();
    private List<EncryptShare> clientShareList = new ArrayList<>();
    private List<EncryptShare> returnShareList = new ArrayList<>();
    private List<String> u1ClientList = new ArrayList<>();
    private List<String> u2UClientList = new ArrayList<>();
    private List<String> u3ClientList = new ArrayList<>();
    private List<DecryptShareSecrets> decryptShareSecretsList = new ArrayList<>();
    private byte[] prime;
    private KEYAgreement keyAgreement = new KEYAgreement();
    private Masking masking = new Masking();
    private ClientListReq clientListReq = new ClientListReq();
    private ReconstructSecretReq reconstructSecretReq = new ReconstructSecretReq();
    private int retCode;

    /**
     * Construct function of cipherClient
     *
     * @param iter         iteration number
     * @param minSecretNum minimum secret shares number used for reconstruct secret
     * @param prime        prime value
     * @param featureSize  featureSize of network
     */
    public CipherClient(int iter, int minSecretNum, byte[] prime, int featureSize) {
        flCommunication = FLCommunication.getInstance();
        this.iteration = iter;
        this.featureSize = featureSize;
        this.minShareNum = minSecretNum;
        this.prime = prime;
    }

    /**
     * Set next request time
     *
     * @param nextRequestTime next request timestamp
     */
    public void setNextRequestTime(String nextRequestTime) {
        this.nextRequestTime = nextRequestTime;
    }

    /**
     * Set client share list
     *
     * @param clientShareList client share list
     */
    private void setClientShareList(List<EncryptShare> clientShareList) {
        this.clientShareList.clear();
        this.clientShareList = clientShareList;
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
     * get retCode
     *
     * @return retCode
     */
    public int getRetCode() {
        return retCode;
    }

    private FLClientStatus genDHKeyPairs() {
        byte[] csk = keyAgreement.generatePrivateKey();
        byte[] cpk = keyAgreement.generatePublicKey(csk);
        if (cpk == null || cpk.length == 0) {
            LOGGER.severe(Common.addTag("[genDHKeyPairs] the return byte[] <cpk> is null, please check!"));
            return FLClientStatus.FAILED;
        }
        byte[] ssk = keyAgreement.generatePrivateKey();
        byte[] spk = keyAgreement.generatePublicKey(ssk);
        if (spk == null || spk.length == 0) {
            LOGGER.severe(Common.addTag("[genDHKeyPairs] the return byte[] <spk> is null, please check!"));
            return FLClientStatus.FAILED;
        }
        this.cKey.clear();
        this.sKey.clear();
        this.cKey.add(cpk);
        this.cKey.add(csk);
        this.sKey.add(spk);
        this.sKey.add(ssk);
        return FLClientStatus.SUCCESS;
    }

    private FLClientStatus genIndividualSecret() {
        byte[] key = new byte[SEED_SIZE];
        int tag = masking.getRandomBytes(key);
        if (tag == -1) {
            LOGGER.severe(Common.addTag("[genIndividualSecret] the return value is -1, please check!"));
            return FLClientStatus.FAILED;
        }
        this.bu = key;
        return FLClientStatus.SUCCESS;
    }

    private List<ShareSecret> genSecretShares(byte[] secret) {
        if (secret == null || secret.length == 0) {
            LOGGER.severe(Common.addTag("[genSecretShares] the input argument <secret> is null"));
            return new ArrayList<>();
        }
        int size = u1ClientList.size();
        if (size <= 1) {
            LOGGER.severe(Common.addTag("[genSecretShares] the size of u1ClientList is not valid: <= 1, it should be " +
                    "> 1"));
            return new ArrayList<>();
        }
        ShareSecrets shamir = new ShareSecrets(minShareNum, size - 1);
        ShareSecrets.SecretShares[] shares = shamir.split(secret, prime);
        if (shares == null || shares.length == 0) {
            LOGGER.severe(Common.addTag("[genSecretShares] the return ShareSecrets.SecretShare[] is null, please " +
                    "check!"));
            return new ArrayList<>();
        }
        int shareIndex = 0;
        List<ShareSecret> shareSecretList = new ArrayList<>();
        for (String vFlID : u1ClientList) {
            if (localFLParameter.getFlID().equals(vFlID)) {
                continue;
            }
            if (shareIndex >= shares.length) {
                LOGGER.severe(Common.addTag("[genSecretShares] the shareIndex is out of range in array <shares>, " +
                        "please check!"));
                return new ArrayList<>();
            }
            int index = shares[shareIndex].getNumber();
            BigInteger intShare = shares[shareIndex].getShares();
            byte[] share = BaseUtil.bigInteger2byteArray(intShare);
            NewArray<byte[]> array = new NewArray<>();
            array.setSize(share.length);
            array.setArray(share);
            ShareSecret shareSecret = new ShareSecret();
            shareSecret.setFlID(vFlID);
            shareSecret.setShare(array);
            shareSecret.setIndex(index);
            shareSecretList.add(shareSecret);
            shareIndex += 1;
        }
        return shareSecretList;
    }

    private FLClientStatus genEncryptExchangedKeys() {
        cUVKeys.clear();
        for (String key : clientPublicKeyList.keySet()) {
            ClientPublicKey curPublicKey = clientPublicKeyList.get(key);
            String vFlID = curPublicKey.getFlID();
            if (localFLParameter.getFlID().equals(vFlID)) {
                continue;
            }
            if (cKey.size() < 2) {
                LOGGER.severe(Common.addTag("[genEncryptExchangedKeys] the size of cKey is not valid: < 2, it should " +
                        "be >= 2, please check!"));
                return FLClientStatus.FAILED;
            }
            byte[] secret1 = keyAgreement.keyAgreement(cKey.get(1), curPublicKey.getCPK().getArray());
            if (secret1 == null || secret1.length == 0) {
                LOGGER.severe(Common.addTag("[genEncryptExchangedKeys] the returned secret1 is null, please check!"));
                return FLClientStatus.FAILED;
            }
            byte[] salt = new byte[0];
            byte[] secret = keyAgreement.getEncryptedPassword(secret1, salt);
            if (secret == null || secret.length == 0) {
                LOGGER.severe(Common.addTag("[genEncryptExchangedKeys] the returned secret is null, please check!"));
                return FLClientStatus.FAILED;
            }
            cUVKeys.put(vFlID, secret);
        }
        return FLClientStatus.SUCCESS;
    }

    private FLClientStatus encryptShares() {
        LOGGER.info(Common.addTag("[PairWiseMask] ************** generate encrypt share secrets for " +
                "RequestShareSecrets **************"));
        // connect sSkUv, bUV, sIndex, indexB  and  then Encrypt them
        if (sKey.size() < 2) {
            LOGGER.severe(Common.addTag("[encryptShares] the size of sKey is not valid: < 2, it should be >= 2, " +
                    "please check!"));
            return FLClientStatus.FAILED;
        }
        List<ShareSecret> sSkUv = genSecretShares(sKey.get(1));
        if (sSkUv.isEmpty()) {
            LOGGER.severe(Common.addTag("[encryptShares] the returned List<ShareSecret> sSkUv is empty, please " +
                    "check!"));
            return FLClientStatus.FAILED;
        }
        List<ShareSecret> bUV = genSecretShares(bu);
        if (sSkUv.isEmpty()) {
            LOGGER.severe(Common.addTag("[encryptShares] the returned List<ShareSecret> bUV is empty, please check!"));
            return FLClientStatus.FAILED;
        }
        if (sSkUv.size() != bUV.size()) {
            LOGGER.severe(Common.addTag("[encryptShares] the sSkUv.size() should be equal to bUV.size(), please " +
                    "check!"));
            return FLClientStatus.FAILED;
        }
        List<EncryptShare> encryptShareList = new ArrayList<>();
        for (int i = 0; i < bUV.size(); i++) {
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
            String vFlID = bUV.get(i).getFlID();
            if (!cUVKeys.containsKey(vFlID)) {
                LOGGER.severe(Common.addTag("[encryptShares] the key " + vFlID + " is not in map cUVKeys, please " +
                        "check!"));
                return FLClientStatus.FAILED;
            }
            AESEncrypt aesEncrypt = new AESEncrypt(cUVKeys.get(vFlID), "CBC");
            byte[] encryptData = aesEncrypt.encrypt(cUVKeys.get(vFlID), allSecret);
            if (encryptData == null || encryptData.length == 0) {
                LOGGER.severe(Common.addTag("[encryptShares] the return byte[] is null, please check!"));
                return FLClientStatus.FAILED;
            }
            NewArray<byte[]> array = new NewArray<>();
            array.setSize(encryptData.length);
            array.setArray(encryptData);
            EncryptShare encryptShare = new EncryptShare();
            encryptShare.setFlID(vFlID);
            encryptShare.setShare(array);
            encryptShareList.add(encryptShare);
        }
        setClientShareList(encryptShareList);
        return FLClientStatus.SUCCESS;
    }

    /**
     * get masked weight of secure aggregation
     *
     * @return masked weight
     */
    public float[] doubleMaskingWeight() {
        List<Float> noiseBu = new ArrayList<>();
        int tag = masking.getMasking(noiseBu, featureSize, bu, individualIv);
        if (tag == -1) {
            LOGGER.severe(Common.addTag("[doubleMaskingWeight] the return value is -1, please check!"));
            return new float[0];
        }
        float[] mask = new float[featureSize];
        for (String vFlID : u2UClientList) {
            if (!clientPublicKeyList.containsKey(vFlID)) {
                LOGGER.severe(Common.addTag("[doubleMaskingWeight] the key " + vFlID + " is not in map " +
                        "clientPublicKeyList, please check!"));
                return new float[0];
            }
            ClientPublicKey curPublicKey = clientPublicKeyList.get(vFlID);
            if (localFLParameter.getFlID().equals(vFlID)) {
                continue;
            }
            byte[] salt;
            byte[] iVec;
            if (vFlID.compareTo(localFLParameter.getFlID()) < 0) {
                salt = curPublicKey.getPwSalt().getArray();
                iVec = curPublicKey.getPwIv().getArray();
            } else {
                salt = this.pwSalt;
                iVec = this.pwIVec;
            }
            if (sKey.size() < 2) {
                LOGGER.severe(Common.addTag("[doubleMaskingWeight] the size of sKey is not valid: < 2, it should be " +
                        ">= 2, please check!"));
                return new float[0];
            }
            byte[] secret1 = keyAgreement.keyAgreement(sKey.get(1), curPublicKey.getSPK().getArray());
            if (secret1 == null || secret1.length == 0) {
                LOGGER.severe(Common.addTag("[doubleMaskingWeight] the returned secret1 is null, please check!"));
                return new float[0];
            }
            byte[] secret = keyAgreement.getEncryptedPassword(secret1, salt);
            if (secret == null || secret.length == 0) {
                LOGGER.severe(Common.addTag("[doubleMaskingWeight] the returned secret is null, please check!"));
                return new float[0];
            }
            sUVKeys.put(vFlID, secret);
            List<Float> noiseSuv = new ArrayList<>();
            tag = masking.getMasking(noiseSuv, featureSize, secret, iVec);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[doubleMaskingWeight] the return value is -1, please check!"));
                return new float[0];
            }
            int sign;
            if (localFLParameter.getFlID().compareTo(vFlID) > 0) {
                sign = 1;
            } else {
                sign = -1;
            }
            for (int maskIndex = 0; maskIndex < noiseSuv.size(); maskIndex++) {
                mask[maskIndex] = mask[maskIndex] + sign * noiseSuv.get(maskIndex);
            }
        }
        for (int maskIndex = 0; maskIndex < noiseBu.size(); maskIndex++) {
            mask[maskIndex] = mask[maskIndex] + noiseBu.get(maskIndex);
        }
        return mask;
    }

    private NewArray<byte[]> byteToArray(ByteBuffer buf, int size) {
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

    private FLClientStatus requestExchangeKeys() {
        LOGGER.info(Common.addTag("[PairWiseMask] ==============request flID: " + localFLParameter.getFlID() +
                "=============="));
        FLClientStatus status = genDHKeyPairs();
        if (status == FLClientStatus.FAILED) {
            LOGGER.severe(Common.addTag("[requestExchangeKeys] the return status is FAILED, please check!"));
            return FLClientStatus.FAILED;
        }
        if (cKey.size() <= 0 || sKey.size() <= 0) {
            LOGGER.severe(Common.addTag("[requestExchangeKeys] the size of cKey or sKey is not valid: <=0."));
            return FLClientStatus.FAILED;
        }
        if (cKey.size() < 2) {
            LOGGER.severe(Common.addTag("[requestExchangeKeys] the size of cKey is not valid: < 2, it should be >= 2," +
                    " please check!"));
            return FLClientStatus.FAILED;
        }
        if (sKey.size() < 2) {
            LOGGER.severe(Common.addTag("[requestExchangeKeys] the size of sKey is not valid: < 2, it should be >= 2," +
                    " please check!"));
            return FLClientStatus.FAILED;
        }
        FlatBufferBuilder fbBuilder = new FlatBufferBuilder();
        byte[] cPK = cKey.get(0);
        byte[] sPK = sKey.get(0);
        int cpk = RequestExchangeKeys.createCPkVector(fbBuilder, cPK);
        int spk = RequestExchangeKeys.createSPkVector(fbBuilder, sPK);

        byte[] indIv = new byte[I_VEC_LEN];
        byte[] pwIv = new byte[I_VEC_LEN];
        byte[] thisPwSalt = new byte[SALT_SIZE];
        SecureRandom secureRandom = Common.getSecureRandom();
        secureRandom.nextBytes(indIv);
        secureRandom.nextBytes(pwIv);
        secureRandom.nextBytes(thisPwSalt);
        this.individualIv = indIv;
        this.pwIVec = pwIv;
        this.pwSalt = thisPwSalt;
        int indIvFbs = RequestExchangeKeys.createIndIvVector(fbBuilder, indIv);
        int pwIvFbs = RequestExchangeKeys.createPwIvVector(fbBuilder, pwIv);
        int pwSaltFbs = RequestExchangeKeys.createPwSaltVector(fbBuilder, thisPwSalt);

        int id = fbBuilder.createString(localFLParameter.getFlID());
        Date date = new Date();
        long timestamp = date.getTime();
        String dateTime = String.valueOf(timestamp);
        int time = fbBuilder.createString(dateTime);

        int exchangeKeysRoot = RequestExchangeKeys.createRequestExchangeKeys(fbBuilder, id, cpk, spk, iteration, time
                , indIvFbs, pwIvFbs, pwSaltFbs);
        fbBuilder.finish(exchangeKeysRoot);
        byte[] msg = fbBuilder.sizedByteArray();
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                flParameter.getDomainName());
        try {
            byte[] responseData = flCommunication.syncRequest(url + "/exchangeKeys", msg);
            if (!Common.isSeverReady(responseData)) {
                LOGGER.info(Common.addTag("[requestExchangeKeys] the server is not ready now, need wait some time and" +
                        " " +
                        "request again"));
                Common.sleep(SLEEP_TIME);
                nextRequestTime = "";
                return FLClientStatus.RESTART;
            }
            ByteBuffer buffer = ByteBuffer.wrap(responseData);
            ResponseExchangeKeys responseExchangeKeys = ResponseExchangeKeys.getRootAsResponseExchangeKeys(buffer);
            return judgeRequestExchangeKeys(responseExchangeKeys);
        } catch (IOException ex) {
            LOGGER.severe(Common.addTag("[requestExchangeKeys] catch IOException: " + ex.getMessage()));
            return FLClientStatus.FAILED;
        }
    }

    private FLClientStatus judgeRequestExchangeKeys(ResponseExchangeKeys bufData) {
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
                LOGGER.info(Common.addTag("[PairWiseMask] RequestExchangeKeys out of time: need wait and request " +
                        "startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch RequestError or SystemError in RequestExchangeKeys"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ResponseExchangeKeys " +
                        "is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    private FLClientStatus getExchangeKeys() {
        FlatBufferBuilder fbBuilder = new FlatBufferBuilder();
        int id = fbBuilder.createString(localFLParameter.getFlID());
        Date date = new Date();
        long timestamp = date.getTime();
        String dateTime = String.valueOf(timestamp);
        int time = fbBuilder.createString(dateTime);
        int getExchangeKeysRoot = GetExchangeKeys.createGetExchangeKeys(fbBuilder, id, iteration, time);
        fbBuilder.finish(getExchangeKeysRoot);
        byte[] msg = fbBuilder.sizedByteArray();
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                flParameter.getDomainName());
        try {
            byte[] responseData = flCommunication.syncRequest(url + "/getKeys", msg);
            if (!Common.isSeverReady(responseData)) {
                LOGGER.info(Common.addTag("[getExchangeKeys] the server is not ready now, need wait some time and " +
                        "request again"));
                Common.sleep(SLEEP_TIME);
                nextRequestTime = "";
                return FLClientStatus.RESTART;
            }
            ByteBuffer buffer = ByteBuffer.wrap(responseData);
            ReturnExchangeKeys returnExchangeKeys = ReturnExchangeKeys.getRootAsReturnExchangeKeys(buffer);
            return judgeGetExchangeKeys(returnExchangeKeys);
        } catch (IOException ex) {
            LOGGER.severe(Common.addTag("[getExchangeKeys] catch IOException: " + ex.getMessage()));
            return FLClientStatus.FAILED;
        }
    }

    private FLClientStatus judgeGetExchangeKeys(ReturnExchangeKeys bufData) {
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
                    ByteBuffer bufPwIv = bufData.remotePublickeys(i).pwIvAsByteBuffer();
                    int sizePwIv = bufData.remotePublickeys(i).pwIvLength();
                    ByteBuffer bufPwSalt = bufData.remotePublickeys(i).pwSaltAsByteBuffer();
                    int sizePwSalt = bufData.remotePublickeys(i).pwSaltLength();
                    publicKey.setCPK(byteToArray(bufCpk, sizeCpk));
                    publicKey.setSPK(byteToArray(bufSpk, sizeSpk));
                    publicKey.setPwIv(byteToArray(bufPwIv, sizePwIv));
                    publicKey.setPwSalt(byteToArray(bufPwSalt, sizePwSalt));
                    clientPublicKeyList.put(srcFlId, publicKey);
                    u1ClientList.add(srcFlId);
                }
                return FLClientStatus.SUCCESS;
            case (ResponseCode.SucNotReady):
                LOGGER.info(Common.addTag("[PairWiseMask] server is not ready now, need wait and request " +
                        "GetExchangeKeys again!"));
                return FLClientStatus.WAIT;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[PairWiseMask] GetExchangeKeys out of time: need wait and request " +
                        "startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch SucNotMatch or SystemError in GetExchangeKeys"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ReturnExchangeKeys is" +
                        " invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    private FLClientStatus requestShareSecrets() {
        FLClientStatus status = genIndividualSecret();
        if (status == FLClientStatus.FAILED) {
            LOGGER.severe(Common.addTag("[requestShareSecrets] the returned status is FAILED from genIndividualSecret" +
                    "(), please check!"));
            return FLClientStatus.FAILED;
        }
        status = genEncryptExchangedKeys();
        if (status == FLClientStatus.FAILED) {
            LOGGER.severe(Common.addTag("[requestShareSecrets] the returned status is FAILED from " +
                    "genEncryptExchangedKeys(), please check!"));
            return FLClientStatus.FAILED;
        }
        status = encryptShares();
        if (status == FLClientStatus.FAILED) {
            LOGGER.severe(Common.addTag("[requestShareSecrets] the returned status is FAILED from encryptShares(), " +
                    "please check!"));
            return FLClientStatus.FAILED;
        }
        FlatBufferBuilder fbBuilder = new FlatBufferBuilder();
        int id = fbBuilder.createString(localFLParameter.getFlID());
        Date date = new Date();
        long timestamp = date.getTime();
        String dateTime = String.valueOf(timestamp);
        int time = fbBuilder.createString(dateTime);
        int clientShareSize = clientShareList.size();
        if (clientShareSize <= 0) {
            LOGGER.warning(Common.addTag("[PairWiseMask] encrypt shares is not ready now!"));
            Common.sleep(SLEEP_TIME);
            return requestShareSecrets();
        } else {
            int[] add = new int[clientShareSize];
            for (int i = 0; i < clientShareSize; i++) {
                int flID = fbBuilder.createString(clientShareList.get(i).getFlID());
                int shareSecretFbs = ClientShare.createShareVector(fbBuilder,
                        clientShareList.get(i).getShare().getArray());
                ClientShare.startClientShare(fbBuilder);
                ClientShare.addFlId(fbBuilder, flID);
                ClientShare.addShare(fbBuilder, shareSecretFbs);
                int clientShareRoot = ClientShare.endClientShare(fbBuilder);
                add[i] = clientShareRoot;
            }
            int encryptedSharesFbs = RequestShareSecrets.createEncryptedSharesVector(fbBuilder, add);
            int requestShareSecretsRoot = RequestShareSecrets.createRequestShareSecrets(fbBuilder, id,
                    encryptedSharesFbs, iteration, time);
            fbBuilder.finish(requestShareSecretsRoot);
            byte[] msg = fbBuilder.sizedByteArray();
            String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                    flParameter.getDomainName());
            try {
                byte[] responseData = flCommunication.syncRequest(url + "/shareSecrets", msg);
                if (!Common.isSeverReady(responseData)) {
                    LOGGER.info(Common.addTag("[requestShareSecrets] the server is not ready now, need wait some time" +
                            " " +
                            "and request again"));
                    Common.sleep(SLEEP_TIME);
                    nextRequestTime = "";
                    return FLClientStatus.RESTART;
                }
                ByteBuffer buffer = ByteBuffer.wrap(responseData);
                ResponseShareSecrets responseShareSecrets = ResponseShareSecrets.getRootAsResponseShareSecrets(buffer);
                return judgeRequestShareSecrets(responseShareSecrets);
            } catch (IOException ex) {
                LOGGER.severe(Common.addTag("[requestShareSecrets] catch IOException: " + ex.getMessage()));
                return FLClientStatus.FAILED;
            }
        }
    }

    private FLClientStatus judgeRequestShareSecrets(ResponseShareSecrets bufData) {
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
                LOGGER.info(Common.addTag("[PairWiseMask] RequestShareSecrets out of time: need wait and request " +
                        "startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch SucNotMatch or SystemError in RequestShareSecrets"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ResponseShareSecrets " +
                        "is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    private FLClientStatus getShareSecrets() {
        FlatBufferBuilder fbBuilder = new FlatBufferBuilder();
        int id = fbBuilder.createString(localFLParameter.getFlID());
        Date date = new Date();
        long timestamp = date.getTime();
        String dateTime = String.valueOf(timestamp);
        int time = fbBuilder.createString(dateTime);
        int getShareSecrets = GetShareSecrets.createGetShareSecrets(fbBuilder, id, iteration, time);
        fbBuilder.finish(getShareSecrets);
        byte[] msg = fbBuilder.sizedByteArray();
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                flParameter.getDomainName());
        try {
            byte[] responseData = flCommunication.syncRequest(url + "/getSecrets", msg);
            if (!Common.isSeverReady(responseData)) {
                LOGGER.info(Common.addTag("[getShareSecrets] the server is not ready now, need wait some time and " +
                        "request again"));
                Common.sleep(SLEEP_TIME);
                nextRequestTime = "";
                return FLClientStatus.RESTART;
            }
            ByteBuffer buffer = ByteBuffer.wrap(responseData);
            ReturnShareSecrets returnShareSecrets = ReturnShareSecrets.getRootAsReturnShareSecrets(buffer);
            return judgeGetShareSecrets(returnShareSecrets);
        } catch (IOException ex) {
            LOGGER.severe(Common.addTag("[getShareSecrets] catch IOException: " + ex.getMessage()));
            return FLClientStatus.FAILED;
        }
    }

    private FLClientStatus judgeGetShareSecrets(ReturnShareSecrets bufData) {
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
                    ClientShare clientShare = bufData.encryptedShares(i);
                    if (clientShare == null) {
                        LOGGER.severe(Common.addTag("[PairWiseMask] the clientShare returned from server is null"));
                        return FLClientStatus.FAILED;
                    }
                    shareSecret.setFlID(clientShare.flId());
                    ByteBuffer bufShare = clientShare.shareAsByteBuffer();
                    int sizeShare = clientShare.shareLength();
                    shareSecret.setShare(byteToArray(bufShare, sizeShare));
                    returnShareList.add(shareSecret);
                    u2UClientList.add(clientShare.flId());
                }
                return FLClientStatus.SUCCESS;
            case (ResponseCode.SucNotReady):
                LOGGER.info(Common.addTag("[PairWiseMask] server is not ready now, need wait and request " +
                        "GetShareSecrets again!"));
                return FLClientStatus.WAIT;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[PairWiseMask] GetShareSecrets out of time: need wait and request " +
                        "startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch SucNotMatch or SystemError in GetShareSecrets"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ReturnShareSecrets is" +
                        " invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    /**
     * exchangeKeys round of secure aggregation
     *
     * @return round execution result
     */
    public FLClientStatus exchangeKeys() {
        LOGGER.info(Common.addTag("[PairWiseMask] ==================== round0: RequestExchangeKeys+GetExchangeKeys " +
                "======================"));
        // RequestExchangeKeys
        FLClientStatus curStatus;
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

    /**
     * shareSecrets round of secure aggregation
     *
     * @return round execution result
     */
    public FLClientStatus shareSecrets() {
        LOGGER.info(Common.addTag(("[PairWiseMask] ==================== round1: RequestShareSecrets+GetShareSecrets " +
                "======================")));
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

    /**
     * reconstructSecrets round of secure aggregation
     *
     * @return round execution result
     */
    public FLClientStatus reconstructSecrets() {
        LOGGER.info(Common.addTag("[PairWiseMask] =================== round3: GetClientList+SendReconstructSecret " +
                "========================"));
        FLClientStatus curStatus;
        // GetClientList
        curStatus = clientListReq.getClientList(iteration, u3ClientList, decryptShareSecretsList, returnShareList,
                cUVKeys);
        while (curStatus == FLClientStatus.WAIT) {
            Common.sleep(SLEEP_TIME);
            curStatus = clientListReq.getClientList(iteration, u3ClientList, decryptShareSecretsList, returnShareList
                    , cUVKeys);
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