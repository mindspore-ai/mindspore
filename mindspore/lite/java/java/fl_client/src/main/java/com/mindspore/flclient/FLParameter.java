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

import java.util.Arrays;
import java.util.UUID;
import java.util.logging.Logger;

/**
 * Defines global parameters used during federated learning and these parameters are provided for users to set.
 *
 * @since 2021-06-30
 */
public class FLParameter {
    private static final Logger LOGGER = Logger.getLogger(FLParameter.class.toString());

    /**
     * The timeout interval for communication on the device.
     */
    public static final int TIME_OUT = 100;

    /**
     * The waiting time of repeated requests.
     */
    public static final int SLEEP_TIME = 1000;
    private static volatile FLParameter flParameter;

    private String domainName;
    private String certPath;
    private String trainDataset;
    private String vocabFile = "null";
    private String idsFile = "null";
    private String testDataset = "null";
    private String flName;
    private String trainModelPath;
    private String inferModelPath;
    private String clientID;
    private boolean useSSL = false;
    private int timeOut;
    private int sleepTime;
    private boolean ifUseElb = false;
    private int serverNum = 1;

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

    public String getDomainName() {
        if (domainName == null || domainName.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <domainName> is null or empty, please set it " +
                    "before use"));
            throw new IllegalArgumentException();
        }
        return domainName;
    }

    public void setDomainName(String domainName) {
        if (domainName == null || domainName.isEmpty() || (!("https".equals(domainName.split(":")[0]) || "http".equals(domainName.split(":")[0])))) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <domainName> is not valid, it should be like " +
                    "as https://...... or http://......, please check it before set"));
            throw new IllegalArgumentException();
        }
        this.domainName = domainName;
    }

    public String getCertPath() {
        if (certPath == null || certPath.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <certPath> is null or empty, please set it " +
                    "before use"));
            throw new IllegalArgumentException();
        }
        return certPath;
    }

    public void setCertPath(String certPath) {
        String realCertPath = Common.getRealPath(certPath);
        if (Common.checkPath(realCertPath)) {
            this.certPath = realCertPath;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <certPath> is not exist, please check it " +
                    "before set"));
            throw new IllegalArgumentException();
        }
    }

    public String getTrainDataset() {
        if (trainDataset == null || trainDataset.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainDataset> is null or empty, please set " +
                    "it before use"));
            throw new IllegalArgumentException();
        }
        return trainDataset;
    }

    public void setTrainDataset(String trainDataset) {
        String realTrainDataset = Common.getRealPath(trainDataset);
        if (Common.checkPath(realTrainDataset)) {
            this.trainDataset = realTrainDataset;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainDataset> is not exist, please check it " +
                    "before set"));
            throw new IllegalArgumentException();
        }
    }

    public String getVocabFile() {
        if ("null".equals(vocabFile) && ALBERT.equals(flName)) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <vocabFile> is null, please set it before " +
                    "use"));
            throw new IllegalArgumentException();
        }
        return vocabFile;
    }

    public void setVocabFile(String vocabFile) {
        String realVocabFile = Common.getRealPath(vocabFile);
        if (Common.checkPath(realVocabFile)) {
            this.vocabFile = realVocabFile;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <vocabFile> is not exist, please check it " +
                    "before set"));
            throw new IllegalArgumentException();
        }
    }

    public String getIdsFile() {
        if ("null".equals(idsFile) && ALBERT.equals(flName)) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <idsFile> is null, please set it before use"));
            throw new IllegalArgumentException();
        }
        return idsFile;
    }

    public void setIdsFile(String idsFile) {
        String realIdsFile = Common.getRealPath(idsFile);
        if (Common.checkPath(realIdsFile)) {
            this.idsFile = realIdsFile;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <idsFile> is not exist, please check it " +
                    "before set"));
            throw new IllegalArgumentException();
        }
    }

    public String getTestDataset() {
        return testDataset;
    }

    public void setTestDataset(String testDataset) {
        String realTestDataset = Common.getRealPath(testDataset);
        if (Common.checkPath(realTestDataset)) {
            this.testDataset = realTestDataset;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <testDataset> is not exist, please check it " +
                    "before set"));
            throw new IllegalArgumentException();
        }
    }

    public String getFlName() {
        if (flName == null || flName.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <flName> is null or empty, please set it " +
                    "before use"));
            throw new IllegalArgumentException();
        }
        return flName;
    }

    public void setFlName(String flName) {
        if (Common.checkFLName(flName)) {
            this.flName = flName;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <flName> is not in FL_NAME_TRUST_LIST: " +
                    Arrays.toString(Common.FL_NAME_TRUST_LIST.toArray(new String[0])) + ", please check it before " +
                    "set"));
            throw new IllegalArgumentException();
        }
    }

    public String getTrainModelPath() {
        if (trainModelPath == null || trainModelPath.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainModelPath> is null or empty, please set" +
                    " it before use"));
            throw new IllegalArgumentException();
        }
        return trainModelPath;
    }

    public void setTrainModelPath(String trainModelPath) {
        String realTrainModelPath = Common.getRealPath(trainModelPath);
        if (Common.checkPath(realTrainModelPath)) {
            this.trainModelPath = realTrainModelPath;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainModelPath> is not exist, please check " +
                    "it before set"));
            throw new IllegalArgumentException();
        }
    }

    public String getInferModelPath() {
        if (inferModelPath == null || inferModelPath.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <inferModelPath> is null or empty, please set" +
                    " it before use"));
            throw new IllegalArgumentException();
        }
        return inferModelPath;
    }

    public void setInferModelPath(String inferModelPath) {
        String realInferModelPath = Common.getRealPath(inferModelPath);
        if (Common.checkPath(realInferModelPath)) {
            this.inferModelPath = realInferModelPath;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <inferModelPath> is not exist, please check " +
                    "it before set"));
            throw new IllegalArgumentException();
        }
    }

    public boolean isUseSSL() {
        return useSSL;
    }

    public void setUseSSL(boolean useSSL) {
        this.useSSL = useSSL;
    }

    public int getTimeOut() {
        return timeOut;
    }

    public void setTimeOut(int timeOut) {
        this.timeOut = timeOut;
    }

    public int getSleepTime() {
        return sleepTime;
    }

    public void setSleepTime(int sleepTime) {
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
                    "set it before use"));
            throw new IllegalArgumentException();
        }
        return serverNum;
    }

    public void setServerNum(int serverNum) {
        this.serverNum = serverNum;
    }

    public String getClientID() {
        if (clientID == null || clientID.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <clientID> is null or empty, please check"));
            throw new IllegalArgumentException();
        }
        return clientID;
    }
}
