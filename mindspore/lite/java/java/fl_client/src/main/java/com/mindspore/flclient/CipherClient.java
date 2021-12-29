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
import com.mindspore.flclient.cipher.CertVerify;
import com.mindspore.flclient.cipher.ClientListReq;
import com.mindspore.flclient.cipher.KEYAgreement;
import com.mindspore.flclient.cipher.Masking;
import com.mindspore.flclient.cipher.ReconstructSecretReq;
import com.mindspore.flclient.cipher.ShareSecrets;
import com.mindspore.flclient.cipher.SignAndVerify;
import com.mindspore.flclient.cipher.struct.ClientPublicKey;
import com.mindspore.flclient.cipher.struct.DecryptShareSecrets;
import com.mindspore.flclient.cipher.struct.EncryptShare;
import com.mindspore.flclient.cipher.struct.NewArray;
import com.mindspore.flclient.cipher.struct.ShareSecret;
import com.mindspore.flclient.pki.PkiUtil;

import mindspore.schema.ClientShare;
import mindspore.schema.GetExchangeKeys;
import mindspore.schema.GetShareSecrets;
import mindspore.schema.RequestAllClientListSign;
import mindspore.schema.RequestExchangeKeys;
import mindspore.schema.RequestShareSecrets;
import mindspore.schema.ResponseClientListSign;
import mindspore.schema.ResponseCode;
import mindspore.schema.ResponseExchangeKeys;
import mindspore.schema.ResponseShareSecrets;
import mindspore.schema.ReturnAllClientListSign;
import mindspore.schema.ReturnExchangeKeys;
import mindspore.schema.ReturnShareSecrets;
import mindspore.schema.SendClientListSign;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.security.cert.CertificateEncodingException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Base64;
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
    private Map<String, X509Certificate[]> certificateList = new HashMap<String, X509Certificate[]>();

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
        FlatBufferBuilder fbBuilder = new FlatBufferBuilder();

        int indIvFbs = RequestExchangeKeys.createIndIvVector(fbBuilder, indIv);
        int pwIvFbs = RequestExchangeKeys.createPwIvVector(fbBuilder, pwIv);
        int pwSaltFbs = RequestExchangeKeys.createPwSaltVector(fbBuilder, thisPwSalt);

        int exchangeKeysRoot;
        byte[] cPK = cKey.get(0);
        byte[] sPK = sKey.get(0);
        int cpk = RequestExchangeKeys.createCPkVector(fbBuilder, cPK);
        int spk = RequestExchangeKeys.createSPkVector(fbBuilder, sPK);
        int id = fbBuilder.createString(localFLParameter.getFlID());
        Date date = new Date();
        long timestamp = date.getTime();
        String dateTime = String.valueOf(timestamp);
        int time = fbBuilder.createString(dateTime);
        String clientID = flParameter.getClientID();

        // for pkiVerify mode
        int certificatesInt = 0;
        int signed = 0;
        if (flParameter.isPkiVerify()) {
            // waiting for certificates to take effect
            int waitTakeEffectTime = 5000;
            Common.sleep(waitTakeEffectTime);
            int nSize = 2;  // exchange equipment certificate and service equipment
            String[] pemCertificateChains = transformX509ArrayToPemArray(CertVerify.getX509CertificateChain(clientID));
            int[] pemList = new int[nSize];
            for (int i = 0; i < nSize; i++) {
                pemList[i] = fbBuilder.createString(pemCertificateChains[i]);
            }
            certificatesInt = RequestExchangeKeys.createCertificateChainVector(fbBuilder, pemList);
            byte[] signature = signPkAndTime(clientID, cPK, sPK, dateTime, iteration);
            signed = RequestExchangeKeys.createSignatureVector(fbBuilder, signature);
        }

        // start build
        RequestExchangeKeys.startRequestExchangeKeys(fbBuilder);
        RequestExchangeKeys.addFlId(fbBuilder, id);
        RequestExchangeKeys.addCPk(fbBuilder, cpk);
        RequestExchangeKeys.addSPk(fbBuilder, spk);
        RequestExchangeKeys.addIteration(fbBuilder, iteration);
        RequestExchangeKeys.addTimestamp(fbBuilder, time);
        RequestExchangeKeys.addIndIv(fbBuilder, indIvFbs);
        RequestExchangeKeys.addPwIv(fbBuilder, pwIvFbs);
        RequestExchangeKeys.addPwSalt(fbBuilder, pwSaltFbs);
        if (flParameter.isPkiVerify()) {
            RequestExchangeKeys.addSignature(fbBuilder, signed);
            RequestExchangeKeys.addCertificateChain(fbBuilder, certificatesInt);
        }
        
        exchangeKeysRoot = RequestExchangeKeys.endRequestExchangeKeys(fbBuilder);
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
                retCode = ResponseCode.OutOfTime;
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

        int getExchangeKeysRoot;
        byte[] signature = signTimeAndIter(dateTime, iteration);
        if (signature == null) {
            LOGGER.severe(Common.addTag("[getExchangeKeys] get signature is null!"));
            return FLClientStatus.FAILED;
        }
        if (signature.length > 0) {
            int signed = GetExchangeKeys.createSignatureVector(fbBuilder, signature);
            // start build
            GetExchangeKeys.startGetExchangeKeys(fbBuilder);
            GetExchangeKeys.addFlId(fbBuilder, id);
            GetExchangeKeys.addIteration(fbBuilder, iteration);
            GetExchangeKeys.addTimestamp(fbBuilder, time);
            GetExchangeKeys.addSignature(fbBuilder, signed);
            getExchangeKeysRoot = GetExchangeKeys.endGetExchangeKeys(fbBuilder);
        } else {
            // start build
            GetExchangeKeys.startGetExchangeKeys(fbBuilder);
            GetExchangeKeys.addFlId(fbBuilder, id);
            GetExchangeKeys.addIteration(fbBuilder, iteration);
            GetExchangeKeys.addTimestamp(fbBuilder, time);
            getExchangeKeysRoot = GetExchangeKeys.endGetExchangeKeys(fbBuilder);
        }

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
                retCode = ResponseCode.OutOfTime;
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
                    ByteBuffer bufCpk = bufData.remotePublickeys(i).cPkAsByteBuffer();
                    ByteBuffer bufSpk = bufData.remotePublickeys(i).sPkAsByteBuffer();
                    int sizeCpk = bufData.remotePublickeys(i).cPkLength();
                    int sizeSpk = bufData.remotePublickeys(i).sPkLength();
                    byte[] bufCpkList = byteBufferToList(bufCpk, sizeCpk);
                    byte[] bufSpkList = byteBufferToList(bufSpk, sizeSpk);
                    // copy bufCpkList and bufSpkList
                    byte[] cPkByte = bufCpkList.clone();
                    byte[] sPkByte = bufSpkList.clone();

                    // check signature
                    boolean isPkiVerify = flParameter.isPkiVerify();
                    if (isPkiVerify) {
                        FLClientStatus checkResult = checkSignature(bufData, i, cPkByte, sPkByte);
                        if (checkResult == FLClientStatus.FAILED) {
                            return FLClientStatus.FAILED;
                        }
                    }

                    ClientPublicKey publicKey = new ClientPublicKey();
                    String srcFlId = bufData.remotePublickeys(i).flId();
                    publicKey.setFlID(srcFlId);
                    ByteBuffer bufPwIv = bufData.remotePublickeys(i).pwIvAsByteBuffer();
                    int sizePwIv = bufData.remotePublickeys(i).pwIvLength();
                    ByteBuffer bufPwSalt = bufData.remotePublickeys(i).pwSaltAsByteBuffer();
                    int sizePwSalt = bufData.remotePublickeys(i).pwSaltLength();
                    publicKey.setPwIv(byteToArray(bufPwIv, sizePwIv));
                    publicKey.setPwSalt(byteToArray(bufPwSalt, sizePwSalt));
                    NewArray<byte[]> bufCpkArray = new NewArray<>();
                    bufCpkArray.setSize(sizeCpk);
                    bufCpkArray.setArray(bufCpkList);
                    NewArray<byte[]> bufSpkArray = new NewArray<>();
                    bufSpkArray.setSize(sizeSpk);
                    bufSpkArray.setArray(bufSpkList);
                    publicKey.setCPK(bufCpkArray);
                    publicKey.setSPK(bufSpkArray);
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

    private FLClientStatus checkSignature(ReturnExchangeKeys bufData, int dataIndex, byte[] cPkByte, byte[] sPkByte) {
        ByteBuffer signature = bufData.remotePublickeys(dataIndex).signatureAsByteBuffer();
        if (signature == null) {
            LOGGER.severe(Common.addTag("[checkSignature] the signature get from server is null, please confirm that pki_verify mode is open at server."));
            return FLClientStatus.FAILED;
        }
        byte[] sigByte = new byte[signature.remaining()];
        signature.get(sigByte);
        int certifyNum = bufData.remotePublickeys(dataIndex).certificateChainLength();
        String[] pemCerts = new String[certifyNum];
        for (int certIndex = 0; certIndex < certifyNum; certIndex++) {
            pemCerts[certIndex] = bufData.remotePublickeys(dataIndex).certificateChain(certIndex);
        }

        X509Certificate[] x509Certificates = CertVerify.transformPemArrayToX509Array(pemCerts);
        if (x509Certificates.length < 2) {
            LOGGER.severe(Common.addTag("the length of x509Certificates is not valid, should be >= 2"));
            return FLClientStatus.FAILED;
        }
        String certificateHash = PkiUtil.genHashFromCer(x509Certificates[1]);
        LOGGER.info(Common.addTag("Get certificate hash success!"));

        // check srcId
        String srcFlId = bufData.remotePublickeys(dataIndex).flId();
        if (certificateHash.equals(srcFlId)) {
            LOGGER.info(Common.addTag("Check flID success and source flID is:" + srcFlId));
        } else {
            LOGGER.severe(Common.addTag("Check flID failed!" + "source flID: " + srcFlId + "Hash ID from certificate:" +
                    " " + certificateHash.equals(srcFlId)));
            return FLClientStatus.FAILED;
        }

        certificateList.put(srcFlId, x509Certificates);
        String timestamp = bufData.remotePublickeys(dataIndex).timestamp();
        String clientID = flParameter.getClientID();
        if (!verifySignature(clientID, x509Certificates, sigByte, cPkByte, sPkByte, timestamp, iteration)) {
            LOGGER.info(Common.addTag("[PairWiseMask] FlID: " + srcFlId +
                    ", signature authentication failed"));
            return FLClientStatus.FAILED;
        } else {
            LOGGER.info(Common.addTag("[PairWiseMask] Verify signature success!"));
        }

        // check iteration and timestamp
        int remoteIter = bufData.iteration();
        FLClientStatus iterTimeCheck = checkIterAndTimestamp(remoteIter, timestamp);
        if (iterTimeCheck == FLClientStatus.FAILED) {
            return FLClientStatus.FAILED;
        }

        return FLClientStatus.SUCCESS;
    }

    private FLClientStatus checkIterAndTimestamp(int remoteIter, String timestamp) {
        if (remoteIter != iteration) {
            LOGGER.severe(Common.addTag("[PairWiseMask] iteration check failed. Remote iteration of client: " + "is "
                    + remoteIter + ", which is not consistent with current iteration:" + iteration));
            return FLClientStatus.FAILED;
        }
        Date date = new Date();
        long currentTimeStamp = date.getTime();
        if (timestamp == null) {
            LOGGER.severe(Common.addTag("[PairWiseMask] Received timeStamp is null,please check it!"));
            return FLClientStatus.FAILED;
        }
        long remoteTimeStamp = Long.parseLong(timestamp);
        long validIterInterval = flParameter.getValidInterval();
        if (Math.abs(currentTimeStamp - remoteTimeStamp) > validIterInterval) {
            LOGGER.severe(Common.addTag("[PairWiseMask] timeStamp check failed! The difference between" +
                    " remote timestamp and current timestamp is beyond valid iteration interval!"));
            return FLClientStatus.FAILED;
        }
        return FLClientStatus.SUCCESS;
    }

    private byte[] byteBufferToList(ByteBuffer buf, int size) {
        byte[] array = new byte[size];
        for (int i = 0; i < size; i++) {
            byte word = buf.get();
            array[i] = word;
        }
        return array;
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

            int requestShareSecretsRoot;
            byte[] signature = signTimeAndIter(dateTime, iteration);
            if (signature == null) {
                LOGGER.severe(Common.addTag("[PairWiseMask] get signature is null!"));
                return FLClientStatus.FAILED;
            }
            if (signature.length > 0) {
                int signed = RequestShareSecrets.createSignatureVector(fbBuilder, signature);
                // start build
                RequestShareSecrets.startRequestShareSecrets(fbBuilder);
                RequestShareSecrets.addFlId(fbBuilder, id);
                RequestShareSecrets.addEncryptedShares(fbBuilder, encryptedSharesFbs);
                RequestShareSecrets.addIteration(fbBuilder, iteration);
                RequestShareSecrets.addTimestamp(fbBuilder, time);
                RequestShareSecrets.addSignature(fbBuilder, signed);
                requestShareSecretsRoot = RequestShareSecrets.endRequestShareSecrets(fbBuilder);
            } else {
                // start build
                RequestShareSecrets.startRequestShareSecrets(fbBuilder);
                RequestShareSecrets.addFlId(fbBuilder, id);
                RequestShareSecrets.addEncryptedShares(fbBuilder, encryptedSharesFbs);
                RequestShareSecrets.addIteration(fbBuilder, iteration);
                RequestShareSecrets.addTimestamp(fbBuilder, time);
                requestShareSecretsRoot = RequestShareSecrets.endRequestShareSecrets(fbBuilder);
            }

            fbBuilder.finish(requestShareSecretsRoot);
            byte[] msg = fbBuilder.sizedByteArray();
            try {
                String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                        flParameter.getDomainName());
                byte[] responseData = flCommunication.syncRequest(url + "/shareSecrets", msg);
                if (!Common.isSeverReady(responseData)) {
                    LOGGER.info(Common.addTag("[requestShareSecrets] the server is not ready now, need wait some time" +
                            " and request again"));
                    Common.sleep(SLEEP_TIME);
                    nextRequestTime = "";
                    retCode = ResponseCode.OutOfTime;
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

        int getShareSecrets;
        byte[] signature = signTimeAndIter(dateTime, iteration);
        if (signature == null) {
            LOGGER.severe(Common.addTag("[getShareSecrets] get signature is null!"));
            return FLClientStatus.FAILED;
        }
        if (signature.length > 0) {
            int signed = GetShareSecrets.createSignatureVector(fbBuilder, signature);
            GetShareSecrets.startGetShareSecrets(fbBuilder);
            GetShareSecrets.addFlId(fbBuilder, id);
            GetShareSecrets.addIteration(fbBuilder, iteration);
            GetShareSecrets.addTimestamp(fbBuilder, time);
            GetShareSecrets.addSignature(fbBuilder, signed);
            getShareSecrets = GetShareSecrets.endGetShareSecrets(fbBuilder);
        } else {
            GetShareSecrets.startGetShareSecrets(fbBuilder);
            GetShareSecrets.addFlId(fbBuilder, id);
            GetShareSecrets.addIteration(fbBuilder, iteration);
            GetShareSecrets.addTimestamp(fbBuilder, time);
            getShareSecrets = GetShareSecrets.endGetShareSecrets(fbBuilder);
        }
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
                retCode = ResponseCode.OutOfTime;
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

        // clientListCheck
        if (flParameter.isPkiVerify()) {
            LOGGER.info(Common.addTag("[PairWiseMask] The mode is pkiVerify mode, start clientList check ..."));
            curStatus = clientListCheck();
            while (curStatus == FLClientStatus.WAIT) {
                Common.sleep(SLEEP_TIME);
                curStatus = clientListCheck();
            }
            if (curStatus != FLClientStatus.SUCCESS) {
                return curStatus;
            }
        }

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

    private byte[] signPkAndTime(String clientID, byte[] cPK, byte[] sPK, String time, int iterNum) {
        // concatenate cPK, sPK and time
        byte[] concatData = concatenateData(cPK, sPK, time, iterNum);
        LOGGER.info("concatenate data success!");
        // signature
        return SignAndVerify.signData(clientID, concatData);
    }

    private static byte[] concatenateData(byte[] cPK, byte[] sPK, String time, int iterNum) {
        // concatenate cPK, sPK and time
        if (time == null) {
            LOGGER.severe(Common.addTag("[concatenateData] input time is null, please check!"));
            throw new IllegalArgumentException();
        }
        byte[] byteTime = time.getBytes(StandardCharsets.UTF_8);
        String iterString = String.valueOf(iterNum);
        byte[] byteIter = iterString.getBytes(StandardCharsets.UTF_8);
        int concatLength = cPK.length + sPK.length + byteTime.length + byteIter.length;
        byte[] concatData = new byte[concatLength];

        int offset = 0;
        System.arraycopy(cPK, 0, concatData, offset, cPK.length);

        offset += cPK.length;
        System.arraycopy(sPK, 0, concatData, offset, sPK.length);

        offset += sPK.length;
        System.arraycopy(byteTime, 0, concatData, offset, byteTime.length);

        offset += byteTime.length;
        System.arraycopy(byteIter, 0, concatData, offset, byteIter.length);
        return concatData;
    }

    private static byte[] concatenateIterAndTime(String time, int iterNum) {
        // concatenate cPK, sPK and time
        byte[] byteTime = time.getBytes(StandardCharsets.UTF_8);
        String iterString = String.valueOf(iterNum);
        byte[] byteIter = iterString.getBytes(StandardCharsets.UTF_8);
        int concatLength = byteTime.length + byteIter.length;
        byte[] concatData = new byte[concatLength];
        int offset = 0;
        System.arraycopy(byteTime, 0, concatData, offset, byteTime.length);
        offset += byteTime.length;
        System.arraycopy(byteIter, 0, concatData, offset, byteIter.length);
        return concatData;
    }

    private static boolean verifySignature(String clientID, X509Certificate[] x509Certificates, byte[] signature,
                                           byte[] cPK, byte[] sPK, String timestamp, int iteration) {
        byte[] concatData = concatenateData(cPK, sPK, timestamp, iteration);
        return SignAndVerify.verifySignatureByCert(clientID, x509Certificates, concatData, signature);
    }

    private FLClientStatus clientListCheck() {
        LOGGER.info(Common.addTag("[PairWiseMask] ==================== ClientListCheck ======================"));
        FLClientStatus curStatus;
        // send signed clientList

        curStatus = sendClientListSign();
        while (curStatus == FLClientStatus.WAIT) {
            Common.sleep(SLEEP_TIME);
            curStatus = sendClientListSign();
        }
        if (curStatus != FLClientStatus.SUCCESS) {
            return curStatus;
        }

        // get signed clientList

        curStatus = getAllClientListSign();
        while (curStatus == FLClientStatus.WAIT) {
            Common.sleep(SLEEP_TIME);
            curStatus = getAllClientListSign();
        }
        return curStatus;
    }

    private FLClientStatus sendClientListSign() {
        LOGGER.info(Common.addTag("[PairWiseMask] ==============request flID: " +
                localFLParameter.getFlID() + "=============="));
        genDHKeyPairs();
        List<String> clientList = u3ClientList;
        int listSize = u3ClientList.size();
        if (listSize == 0) {
            LOGGER.severe("[Encrypt] u3List is empty, please check!");
            return FLClientStatus.FAILED;
        }

        // send signature
        byte[] clientListByte = transStringListToByte(clientList);
        byte[] listHash = SignAndVerify.getSHA256(clientListByte);
        String clientID = flParameter.getClientID();
        byte[] signature = SignAndVerify.signData(clientID, listHash);
        if (signature == null) {
            LOGGER.severe(Common.addTag("[sendClientListSign] the returned signature is null"));
            return FLClientStatus.FAILED;
        }
        FlatBufferBuilder fbBuilder = new FlatBufferBuilder();
        int signed = RequestExchangeKeys.createSignatureVector(fbBuilder, signature);

        int sendClientListRoot;
        Date date = new Date();
        long timestamp = date.getTime();
        String dateTime = String.valueOf(timestamp);
        byte[] reqSign = signTimeAndIter(dateTime, iteration);
        String flID = localFLParameter.getFlID();
        int id = fbBuilder.createString(flID);
        int time = fbBuilder.createString(dateTime);
        if (signature.length > 0) {
            int reqSigned = SendClientListSign.createSignatureVector(fbBuilder, reqSign);
            sendClientListRoot = SendClientListSign.createSendClientListSign(fbBuilder, id, iteration, time, signed,
                    reqSigned);
        } else {
            sendClientListRoot = SendClientListSign.createSendClientListSign(fbBuilder, id, iteration, time, signed, 0);
        }

        fbBuilder.finish(sendClientListRoot);
        byte[] msg = fbBuilder.sizedByteArray();
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                flParameter.getDomainName());
        try {
            byte[] responseData = flCommunication.syncRequest(url + "/pushListSign", msg);

            if (!Common.isSeverReady(responseData)) {
                LOGGER.info(Common.addTag("[sendClientListSign] the server is not ready now, need wait some time and " +
                        "request again"));
                Common.sleep(SLEEP_TIME);
                nextRequestTime = "";
                retCode = ResponseCode.OutOfTime;
                return FLClientStatus.RESTART;
            }
            ByteBuffer buffer = ByteBuffer.wrap(responseData);
            ResponseClientListSign responseClientListSign =
                    ResponseClientListSign.getRootAsResponseClientListSign(buffer);
            return judgeRequestClientList(responseClientListSign);
        } catch (IOException e) {
            e.printStackTrace();
            return FLClientStatus.FAILED;
        }
    }

    private FLClientStatus judgeRequestClientList(ResponseClientListSign bufData) {
        retCode = bufData.retcode();
        LOGGER.info(Common.addTag("[PairWiseMask] **************the response of RequestClientListSign**************"));
        LOGGER.info(Common.addTag("[PairWiseMask] return code: " + retCode));
        LOGGER.info(Common.addTag("[PairWiseMask] reason: " + bufData.reason()));
        LOGGER.info(Common.addTag("[PairWiseMask] current iteration in server: " + bufData.iteration()));
        LOGGER.info(Common.addTag("[PairWiseMask] next request time: " + bufData.nextReqTime()));
        switch (retCode) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[PairWiseMask] RequestClientListSign success"));
                return FLClientStatus.SUCCESS;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[PairWiseMask] RequestClientListSign out of time: need wait and request " +
                        "startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch RequestError or SystemError in RequestClientListSign"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in RequestClientListSign" +
                        " is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    private FLClientStatus getAllClientListSign() {
        FlatBufferBuilder fbBuilder = new FlatBufferBuilder();
        int id = fbBuilder.createString(localFLParameter.getFlID());
        Date date = new Date();
        long timestamp = date.getTime();
        String dateTime = String.valueOf(timestamp);
        int time = fbBuilder.createString(dateTime);

        int requestAllClientListSign;
        byte[] signature = signTimeAndIter(dateTime, iteration);
        if (signature.length > 0) {
            int signed = RequestAllClientListSign.createSignatureVector(fbBuilder, signature);
            requestAllClientListSign = RequestAllClientListSign.createRequestAllClientListSign(fbBuilder, id,
                    iteration, time, signed);
        } else {
            requestAllClientListSign = RequestAllClientListSign.createRequestAllClientListSign(fbBuilder, id,
                    iteration, time, 0);
        }

        fbBuilder.finish(requestAllClientListSign);
        byte[] msg = fbBuilder.sizedByteArray();
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                flParameter.getDomainName());
        try {
            byte[] responseData = flCommunication.syncRequest(url + "/getListSign", msg);

            if (!Common.isSeverReady(responseData)) {
                LOGGER.info(Common.addTag("[getAllClientListSign] the server is not ready now, need wait some time " +
                        "and request again"));
                Common.sleep(SLEEP_TIME);
                nextRequestTime = "";
                retCode = ResponseCode.OutOfTime;
                return FLClientStatus.RESTART;
            }
            ByteBuffer buffer = ByteBuffer.wrap(responseData);
            ReturnAllClientListSign returnAllClientList =
                    ReturnAllClientListSign.getRootAsReturnAllClientListSign(buffer);
            return judgeAllClientList(returnAllClientList);
        } catch (IOException e) {
            e.printStackTrace();
            return FLClientStatus.FAILED;
        }
    }

    private FLClientStatus judgeAllClientList(ReturnAllClientListSign bufData) {
        retCode = bufData.retcode();
        LOGGER.info(Common.addTag("[PairWiseMask] **************the response of GetAllClientsList**************"));
        LOGGER.info(Common.addTag("[PairWiseMask] return code: " + retCode));
        LOGGER.info(Common.addTag("[PairWiseMask] reason: " + bufData.reason()));
        LOGGER.info(Common.addTag("[PairWiseMask] current iteration in server: " + bufData.iteration()));
        LOGGER.info(Common.addTag("[PairWiseMask] next request time: " + bufData.nextReqTime()));
        switch (retCode) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[PairWiseMask] GetAllClientList success"));
                int length = bufData.clientListSignLength();
                String clientID = flParameter.getClientID();
                String localFlID = localFLParameter.getFlID();
                byte[] localClientList = transStringListToByte(u3ClientList);
                byte[] localListHash = SignAndVerify.getSHA256(localClientList);
                for (int i = 0; i < length; i++) {
                    // verify signature
                    ByteBuffer signature = bufData.clientListSign(i).signatureAsByteBuffer();
                    byte[] sigByte = new byte[signature.remaining()];
                    signature.get(sigByte);
                    if (bufData.clientListSign(i).flId() == null) {
                        LOGGER.severe(Common.addTag("[PairWiseMask] get flID failed!"));
                        return FLClientStatus.FAILED;
                    }
                    String srcFlId = bufData.clientListSign(i).flId();
                    X509Certificate[] remoteCertificates = certificateList.get(srcFlId);
                    if (localFlID.equals(srcFlId)) {
                        continue;
                    }  // Do not verify itself
                    if (!SignAndVerify.verifySignatureByCert(clientID, remoteCertificates, localListHash, sigByte)) {
                        LOGGER.info(Common.addTag("[PairWiseMask] FlID: " + srcFlId +
                                ", signature authentication failed"));
                        return FLClientStatus.FAILED;
                    }
                }
                return FLClientStatus.SUCCESS;
            case (ResponseCode.SucNotReady):
                LOGGER.info(Common.addTag("[PairWiseMask] server is not ready now, need wait and request " +
                        "GetAllClientsList again!"));
                return FLClientStatus.WAIT;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[PairWiseMask] GetAllClientsList out of time: need wait and request " +
                        "startFLJob again"));
                setNextRequestTime(bufData.nextReqTime());
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[PairWiseMask] catch SucNotMatch or SystemError in GetAllClientsList"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[PairWiseMask] the return <retCode> from server in ReturnAllClientList " +
                        "is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    private byte[] transStringListToByte(List<String> stringList) {
        int byteNum = 0;
        for (String value : stringList) {
            byte[] stringByte = value.getBytes(StandardCharsets.UTF_8);
            byteNum += stringByte.length;
        }
        byte[] concatData = new byte[byteNum];
        int offset = 0;
        for (String str : stringList) {
            byte[] stringByte = str.getBytes(StandardCharsets.UTF_8);
            System.arraycopy(stringByte, 0, concatData, offset, stringByte.length);
            offset += stringByte.length;
        }
        return concatData;
    }

    /**
     * Add signature on timestamp and iteration
     *
     * @param dateTime  the timestamp of data
     * @param iteration iteration number
     * @return signed time and iteration
     */
    public static byte[] signTimeAndIter(String dateTime, int iteration) {
        // signature
        FLParameter flParameter = FLParameter.getInstance();
        String clientID = flParameter.getClientID();
        boolean isPkiVerify = flParameter.isPkiVerify();
        byte[] signature = new byte[0];
        if (isPkiVerify) {
            LOGGER.info(Common.addTag("ClientID is:" + clientID));
            byte[] concatData = concatenateIterAndTime(dateTime, iteration);
            signature = SignAndVerify.signData(clientID, concatData);
        }
        return signature;
    }

    private static String transformX509ToPem(X509Certificate x509Certificate) {
        if (x509Certificate == null) {
            LOGGER.severe(Common.addTag("[CertVerify] x509Certificate is null, please check!"));
            return null;
        }
        String pemCert;
        try {
            byte[] derCert = x509Certificate.getEncoded();
            pemCert = new String(Base64.getEncoder().encode(derCert));
        } catch (CertificateEncodingException e) {
            LOGGER.severe(Common.addTag("[CertVerify] catch Exception: " + e.getMessage()));
            return null;
        }
        return pemCert;
    }

    private static String[] transformX509ArrayToPemArray(X509Certificate[] x509Certificates) {
        if (x509Certificates == null || x509Certificates.length == 0) {
            LOGGER.severe(Common.addTag("[CertVerify] certificateChains is null or empty, please check!"));
            throw new IllegalArgumentException();
        }
        int nSize = x509Certificates.length;
        String[] pemCerts = new String[nSize];
        for (int i = 0; i < nSize; ++i) {
            String pemCert = transformX509ToPem(x509Certificates[i]);
            pemCerts[i] = pemCert;
        }
        return pemCerts;
    }
}