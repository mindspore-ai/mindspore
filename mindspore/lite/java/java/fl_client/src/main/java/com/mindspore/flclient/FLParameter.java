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

import static com.mindspore.flclient.LocalFLParameter.ALBERT;

import com.mindspore.flclient.compression.CompressMode;
import com.mindspore.flclient.model.RunType;
import mindspore.schema.CompressType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.logging.Logger;

import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.X509TrustManager;

/**
 * Defines global parameters used during federated learning and these parameters are provided for users to set.
 *
 * @since 2021-06-30
 */
public class FLParameter {
    private static final Logger LOGGER = Logger.getLogger(FLParameter.class.toString());

    /**
     * The timeout interval for communication on the device, time unit: seconds.
     */
    public static final int TIME_OUT = 100;

    /**
     * The waiting time of repeated requests, time unit: milliseconds.
     */
    public static final int SLEEP_TIME = 10000;

    /**
     * The max waiting time when call sleeping, time unit: milliseconds.
     */
    public static final int MAX_SLEEP_TIME = 1800000;

    /**
     * Maximum number of times to repeat RESTART
     */
    public static final int RESTART_TIME_PER_ITER = 1;

    /**
     * Maximum number of times to wait some time and then repeat the same request.
     */
    public static final int MAX_WAIT_TRY_TIME = 18;

    private static volatile FLParameter flParameter;

    private String deployEnv;
    private String domainName;
    private String certPath;

    private SSLSocketFactory sslSocketFactory;
    private X509TrustManager x509TrustManager;
    private IFLJobResultCallback iflJobResultCallback = new FLJobResultCallback();

    private String trainDataset;
    private String vocabFile = "null";
    private String idsFile = "null";
    private String testDataset = "null";
    private boolean useSSL = false;
    private String flName;
    private String trainModelPath;
    private String inferModelPath;
    private String sslProtocol = "TLSv1.2";
    private String clientID;
    private int timeOut;
    private int sleepTime;
    private boolean ifUseElb = false;
    private int serverNum = 1;
    private boolean ifPkiVerify = false;
    private String equipCrlPath = "null";
    private long validIterInterval = 600000L;
    private int threadNum = 1;
    private BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;

    private List<String> trainWeightName = new ArrayList<>();
    private List<String> inferWeightName = new ArrayList<>();
    private Map<RunType, List<String>> dataMap = new HashMap<>();
    private ServerMod serverMod;
    private int batchSize;
    private int[][] inputShape;

    private FLParameter() {
        clientID = UUID.randomUUID().toString();
    }

    /**
     * Get the singleton object of the class FLParameter.
     *
     * @return the singleton object of the class FLParameter.
     */
    public static FLParameter getInstance() {
        FLParameter localRef = flParameter;
        if (localRef == null) {
            synchronized (FLParameter.class) {
                localRef = flParameter;
                if (localRef == null) {
                    flParameter = localRef = new FLParameter();
                }
            }
        }
        return localRef;
    }

    public String getDeployEnv() {
        if (deployEnv == null || deployEnv.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <env> is null, please set it before using"));
            throw new IllegalArgumentException();
        }
        return deployEnv;
    }

    public void setDeployEnv(String env) {
        if (Common.checkEnv(env)) {
            this.deployEnv = env;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <env> is not in envTrustList: x86, android, " +
                    "please check it before setting"));
            throw new IllegalArgumentException();
        }
    }

    public String getDomainName() {
        if (domainName == null || domainName.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <domainName> is null or empty, please set it " +
                    "before using"));
            throw new IllegalArgumentException();
        }
        return domainName;
    }

    public void setDomainName(String domainName) {
        if (domainName == null || domainName.isEmpty() || (!("https".equals(domainName.split(":")[0]) || "http".equals(domainName.split(":")[0])))) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <domainName> is not valid, it should be like " +
                    "as https://...... or http://......, please check it before setting"));
            throw new IllegalArgumentException();
        }
        this.domainName = domainName;
        Common.setIsHttps(domainName.split("//")[0].split(":")[0]);
    }

    public String getClientID() {
        if (clientID == null || clientID.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <clientID> is null or empty, please check"));
            throw new IllegalArgumentException();
        }
        return clientID;
    }

    public void setClientID(String clientID) {
        if (clientID == null || clientID.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <clientID> is null or empty, please check " +
                    "before setting"));
            throw new IllegalArgumentException();
        }
        this.clientID = clientID;
    }

    public String getCertPath() {
        if (certPath == null || certPath.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <certPath> is null or empty, the <certPath> " +
                    "must be set when conducting https communication, please set it by FLParameter.setCertPath()"));
            throw new IllegalArgumentException();
        }
        return certPath;
    }

    public void setCertPath(String certPath) {
        String realCertPath = Common.getRealPath(certPath);
        if (Common.checkPath(realCertPath)) {
            this.certPath = realCertPath;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <certPath> does not exist, it must be a valid" +
                    " path when conducting https communication, please check it before setting"));
            throw new IllegalArgumentException();
        }
    }

    public SSLSocketFactory getSslSocketFactory() {
        if (sslSocketFactory == null) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <sslSocketFactory> is null, the " +
                    "<sslSocketFactory> must be set when the deployEnv being \"android\", please set it by " +
                    "FLParameter.setSslSocketFactory()"));
            throw new IllegalArgumentException();
        }
        return sslSocketFactory;
    }

    public void setSslSocketFactory(SSLSocketFactory sslSocketFactory) {
        this.sslSocketFactory = sslSocketFactory;
    }

    public X509TrustManager getX509TrustManager() {
        if (x509TrustManager == null) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <x509TrustManager> is null, the " +
                    "<x509TrustManager> must be set when the deployEnv being \"android\", please set it by " +
                    "FLParameter.setX509TrustManager()"));
            throw new IllegalArgumentException();
        }
        return x509TrustManager;
    }

    public void setX509TrustManager(X509TrustManager x509TrustManager) {
        this.x509TrustManager = x509TrustManager;
    }

    public IFLJobResultCallback getIflJobResultCallback() {
        if (iflJobResultCallback == null) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <iflJobResultCallback> is null, please set it" +
                    " before using"));
            throw new IllegalArgumentException();
        }
        return iflJobResultCallback;
    }

    public void setIflJobResultCallback(IFLJobResultCallback iflJobResultCallback) {
        this.iflJobResultCallback = iflJobResultCallback;
    }

    public String getTrainDataset() {
        if (trainDataset == null || trainDataset.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainDataset> is null or empty, please set " +
                    "it before using"));
            throw new IllegalArgumentException();
        }
        return trainDataset;
    }

    public void setTrainDataset(String trainDataset) {
        LOGGER.warning(Common.addTag(Common.LOG_DEPRECATED));
        String realTrainDataset = Common.getRealPath(trainDataset);
        if (Common.checkPath(realTrainDataset)) {
            this.trainDataset = realTrainDataset;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainDataset> does not exist, please check " +
                    "it before setting"));
            throw new IllegalArgumentException();
        }
    }

    public String getVocabFile() {
        if ("null".equals(vocabFile) && ALBERT.equals(flName)) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <vocabFile> is null, please set it before " +
                    "using"));
            throw new IllegalArgumentException();
        }
        return vocabFile;
    }

    public void setVocabFile(String vocabFile) {
        LOGGER.warning(Common.addTag(Common.LOG_DEPRECATED));
        String realVocabFile = Common.getRealPath(vocabFile);
        if (Common.checkPath(realVocabFile)) {
            this.vocabFile = realVocabFile;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <vocabFile> does not exist, please check it " +
                    "before setting"));
            throw new IllegalArgumentException();
        }
    }

    public String getIdsFile() {
        if ("null".equals(idsFile) && ALBERT.equals(flName)) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <idsFile> is null, please set it before " +
                    "using"));
            throw new IllegalArgumentException();
        }
        return idsFile;
    }

    public void setIdsFile(String idsFile) {
        LOGGER.warning(Common.addTag(Common.LOG_DEPRECATED));
        String realIdsFile = Common.getRealPath(idsFile);
        if (Common.checkPath(realIdsFile)) {
            this.idsFile = realIdsFile;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <idsFile> does not exist, please check it " +
                    "before setting"));
            throw new IllegalArgumentException();
        }
    }

    public String getTestDataset() {
        return testDataset;
    }

    public void setTestDataset(String testDataset) {
        LOGGER.warning(Common.addTag(Common.LOG_DEPRECATED));
        String realTestDataset = Common.getRealPath(testDataset);
        if (Common.checkPath(realTestDataset)) {
            this.testDataset = realTestDataset;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <testDataset> does not exist, please check it" +
                    " before setting"));
            throw new IllegalArgumentException();
        }
    }

    public String getFlName() {
        if (flName == null || flName.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <flName> is null or empty, please set it " +
                    "before using"));
            throw new IllegalArgumentException();
        }
        return flName;
    }

    public void setFlName(String flName) {
        this.flName = flName;
    }

    public String getTrainModelPath() {
        if (trainModelPath == null || trainModelPath.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainModelPath> is null or empty, please set" +
                    " it before using"));
            throw new IllegalArgumentException();
        }
        return trainModelPath;
    }

    public void setTrainModelPath(String trainModelPath) {
        String realTrainModelPath = Common.getRealPath(trainModelPath);
        if (Common.checkPath(realTrainModelPath)) {
            this.trainModelPath = realTrainModelPath;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainModelPath> does not exist, please " +
                    "check it before setting"));
            throw new IllegalArgumentException();
        }
    }

    public String getInferModelPath() {
        if (inferModelPath == null || inferModelPath.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <inferModelPath> is null or empty, please set" +
                    " it before using"));
            throw new IllegalArgumentException();
        }
        return inferModelPath;
    }

    public void setInferModelPath(String inferModelPath) {
        String realInferModelPath = Common.getRealPath(inferModelPath);
        if (Common.checkPath(realInferModelPath)) {
            this.inferModelPath = realInferModelPath;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <inferModelPath> does not exist, please check" +
                    " it before setting"));
            throw new IllegalArgumentException();
        }
    }

    public boolean isUseSSL() {
        return useSSL;
    }

    public void setUseSSL(boolean useSSL) {
        LOGGER.warning(Common.addTag("Certificate authentication is required for https communication, this parameter " +
                "is true by default and no need to set it, " + Common.LOG_DEPRECATED));
        this.useSSL = useSSL;
    }

    public String getSslProtocol() {
        if (sslProtocol == null || sslProtocol.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <sslProtocol> is null or empty, please set it" +
                    " before using"));
            throw new IllegalArgumentException();
        }
        return sslProtocol;
    }

    public void setSslProtocol(String sslProtocol) {
        if (Common.checkSSLProtocol(sslProtocol)) {
            this.sslProtocol = sslProtocol;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <sslProtocol> is not in sslProtocolTrustList " +
                    ": " + Arrays.toString(Common.SSL_PROTOCOL_TRUST_LIST.toArray(new String[0])) + ", please check " +
                    "it before setting"));
            throw new IllegalArgumentException();
        }
    }

    public int getTimeOut() {
        return timeOut;
    }

    public void setTimeOut(int timeOut) {
        if (timeOut <= 0) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <timeOut> <= 0, it should be > 0, please " +
                    "set it before using"));
            throw new IllegalArgumentException();
        }
        this.timeOut = timeOut;
    }

    public int getSleepTime() {
        return sleepTime;
    }

    public void setSleepTime(int sleepTime) {
        if (sleepTime <= 0) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <sleepTime> <= 0, it should be > 0, please " +
                    "set it before using"));
            throw new IllegalArgumentException();
        }
        this.sleepTime = sleepTime;
    }

    public boolean isUseElb() {
        return ifUseElb;
    }

    public void setUseElb(boolean ifUseElb) {
        this.ifUseElb = ifUseElb;
    }

    public int getServerNum() {
        if (serverNum <= 0) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <serverNum> <= 0, it should be > 0, please " +
                    "set it before using"));
            throw new IllegalArgumentException();
        }
        return serverNum;
    }

    public void setServerNum(int serverNum) {
        this.serverNum = serverNum;
    }

    public boolean isPkiVerify() {
        return ifPkiVerify;
    }

    public void setPkiVerify(boolean ifPkiVerify) {
        this.ifPkiVerify = ifPkiVerify;
    }

    public String getEquipCrlPath() {
        if (equipCrlPath == null || equipCrlPath.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <equipCrlPath> is null or empty, please set " +
                    "it before using"));
            throw new IllegalArgumentException();
        }
        return equipCrlPath;
    }

    /**
     * Obtains the Valid Iteration Interval set by a user.
     *
     * @return the Valid Iteration Interval.
     */
    public long getValidInterval() {
        if (validIterInterval <= 0) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <validIterInterval> is not valid, please set " +
                    "it as larger than 0."));
            throw new IllegalArgumentException();
        }
        return validIterInterval;
    }

    public void setEquipCrlPath(String certPath) {
        String realCertPath = Common.getRealPath(certPath);
        if (Common.checkPath(realCertPath)) {
            this.equipCrlPath = realCertPath;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <equipCrlPath> does not exist, please check " +
                    "it before setting"));
            throw new IllegalArgumentException();
        }
    }

    /**
     * Set the Valid Iteration Interval.
     *
     * @param validInterval the Valid Iteration Interval.
     */
    public void setValidInterval(long validInterval) {
        if (validInterval > 0) {
            this.validIterInterval = validInterval;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <validIterInterval> should be larger than 0, " +
                    "please set it again."));
            throw new IllegalArgumentException();
        }
    }

    public int getThreadNum() {
        return threadNum;
    }

    public void setThreadNum(int threadNum) {
        if (threadNum <= 0) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <threadNum> <= 0, please check it before " +
                    "setting"));
            throw new IllegalArgumentException();
        }
        this.threadNum = threadNum;
    }

    public int getCpuBindMode() {
        LOGGER.info(Common.addTag("[flParameter] the parameter of <cpuBindMode> is: " + cpuBindMode.toString() + " , " +
                "the NOT_BINDING_CORE means that not binding core, BIND_LARGE_CORE means binding the large core, " +
                "BIND_MIDDLE_CORE means binding the middle core"));
        return cpuBindMode.ordinal();
    }

    public void setCpuBindMode(BindMode cpuBindMode) {
        this.cpuBindMode = cpuBindMode;
    }

    public void setHybridWeightName(List<String> hybridWeightName, RunType runType) {
        if (RunType.TRAINMODE.equals(runType)) {
            this.trainWeightName = hybridWeightName;
        } else if (RunType.INFERMODE.equals(runType)) {
            this.inferWeightName = hybridWeightName;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the variable <runType> can only be set to <RunType.TRAINMODE> " +
                    "or <RunType.INFERMODE>, please check it"));
            throw new IllegalArgumentException();
        }

    }

    public List<String> getHybridWeightName(RunType runType) {
        if (RunType.TRAINMODE.equals(runType)) {
            if (trainWeightName.isEmpty()) {
                LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainWeightName> is null, please " +
                        "set it before use"));
                throw new IllegalArgumentException();
            }
            return trainWeightName;
        } else if (RunType.INFERMODE.equals(runType)) {
            if (inferWeightName.isEmpty()) {
                LOGGER.severe(Common.addTag("[flParameter] the parameter of <inferWeightName> is null, please " +
                        "set it before use"));
                throw new IllegalArgumentException();
            }
            return inferWeightName;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the variable <runType> can only be set to <RunType.TRAINMODE> " +
                    "or <RunType.INFERMODE>, please check it"));
            throw new IllegalArgumentException();
        }

    }

    public Map<RunType, List<String>> getDataMap() {
        if (dataMap.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <dataMaps> is null, please " +
                    "set it before use"));
            throw new IllegalArgumentException();
        }
        return dataMap;
    }

    public void setDataMap(Map<RunType, List<String>> dataMap) {
        this.dataMap = dataMap;
    }

    public ServerMod getServerMod() {
        if (serverMod == null) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <serverMod> is null, please " +
                    "set it before use"));
            throw new IllegalArgumentException();
        }
        return serverMod;
    }

    public void setServerMod(ServerMod serverMod) {
        this.serverMod = serverMod;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        if (batchSize <= 0) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <batchSize> <= 0, please check it before " +
                    "setting"));
            throw new IllegalArgumentException();
        }
        this.batchSize = batchSize;
    }

    public byte[] getDownloadCompressTypes() {
        byte[] downloadCompressTypes = new byte[CompressMode.COMPRESS_TYPE_MAP.size()];
        int index = 0;
        for (byte downloadCompressType : CompressMode.COMPRESS_TYPE_MAP.keySet()) {
            downloadCompressTypes[index] = downloadCompressType;
            index += 1;
        }
        return downloadCompressTypes;
    }

    public int[][] getInputShape() {
        return inputShape;
    }

    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape;
    }
}
