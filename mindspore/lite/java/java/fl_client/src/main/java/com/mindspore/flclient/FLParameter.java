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

import java.util.logging.Logger;

public class FLParameter {
    private static final Logger LOGGER = Logger.getLogger(FLParameter.class.toString());

    public static final int TIME_OUT = 100;
    public static final int SLEEP_TIME = 1000;

    private String hostName;
    private String certPath;
    private boolean useHttps = false;

    private String trainDataset;
    private String vocabFile = "null";
    private String idsFile = "null";
    private String testDataset = "null";
    private String flName;
    private String trainModelPath;
    private String inferModelPath;
    private String clientID;
    private String ip;
    private int port;
    private boolean useSSL = false;
    private int timeOut;
    private int sleepTime;
    private boolean useElb = false;
    private int serverNum = 1;

    private boolean timer = true;
    private int timeWindow = 6000;
    private int reRequestNum = timeWindow / SLEEP_TIME + 1;

    private static volatile FLParameter flParameter;

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

    public String getHostName() {
        if ("".equals(hostName) || hostName.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <hostName> is null, please set it before use"));
            throw new RuntimeException();
        }
        return hostName;
    }

    public void setHostName(String hostName) {
        this.hostName = hostName;
    }

    public String getCertPath() {
        if ("".equals(certPath) || certPath.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <certPath> is null, please set it before use"));
            throw new RuntimeException();
        }
        return certPath;
    }

    public void setCertPath(String certPath) {
        certPath = Common.getRealPath(certPath);
        if (Common.checkPath(certPath)) {
            this.certPath = certPath;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <certPath> is not exist, please check it before set"));
            throw new RuntimeException();
        }
    }

    public boolean isUseHttps() {
        return useHttps;
    }

    public void setUseHttps(boolean useHttps) {
        this.useHttps = useHttps;
    }

    public String getTrainDataset() {
        if ("".equals(trainDataset) || trainDataset.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainDataset> is null, please set it before use"));
            throw new RuntimeException();
        }
        return trainDataset;
    }

    public void setTrainDataset(String trainDataset) {
        trainDataset = Common.getRealPath(trainDataset);
        if (Common.checkPath(trainDataset)) {
            this.trainDataset = trainDataset;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainDataset> is not exist, please check it before set"));
            throw new RuntimeException();
        }
    }

    public String getVocabFile() {
        if ("null".equals(vocabFile) && "adbert".equals(flName)) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <vocabFile> is null, please set it before use"));
            throw new RuntimeException();
        }
        return vocabFile;
    }

    public void setVocabFile(String vocabFile) {
        vocabFile = Common.getRealPath(vocabFile);
        if (Common.checkPath(vocabFile)) {
            this.vocabFile = vocabFile;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <vocabFile> is not exist, please check it before set"));
            throw new RuntimeException();
        }
    }

    public String getIdsFile() {
        if ("null".equals(idsFile) && "adbert".equals(flName)) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <idsFile> is null, please set it before use"));
            throw new RuntimeException();
        }
        return idsFile;
    }

    public void setIdsFile(String idsFile) {
        idsFile = Common.getRealPath(idsFile);
        if (Common.checkPath(idsFile)) {
            this.idsFile = idsFile;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <idsFile> is not exist, please check it before set"));
            throw new RuntimeException();
        }
    }

    public String getTestDataset() {
        return testDataset;
    }

    public void setTestDataset(String testDataset) {
        testDataset = Common.getRealPath(testDataset);
        if (Common.checkPath(testDataset)) {
            this.testDataset = testDataset;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <testDataset> is not exist, please check it before set"));
            throw new RuntimeException();
        }
    }

    public String getFlName() {
        if ("".equals(flName) || flName.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <flName> is null, please set it before use"));
            throw new RuntimeException();
        }
        return flName;
    }

    public void setFlName(String flName) {
        if (Common.checkFLName(flName)) {
            this.flName = flName;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <flName> is not in flNameTrustList, please check it before set"));
            throw new RuntimeException();
        }
    }

    public String getTrainModelPath() {
        if ("".equals(trainModelPath) || trainModelPath.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainModelPath> is null, please set it before use"));
            throw new RuntimeException();
        }
        return trainModelPath;
    }

    public void setTrainModelPath(String trainModelPath) {
        trainModelPath = Common.getRealPath(trainModelPath);
        if (Common.checkPath(trainModelPath)) {
            this.trainModelPath = trainModelPath;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <trainModelPath> is not exist, please check it before set"));
            throw new RuntimeException();
        }
    }

    public String getInferModelPath() {
        if ("".equals(inferModelPath) || inferModelPath.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <inferModelPath> is null, please set it before use"));
            throw new RuntimeException();
        }
        return inferModelPath;
    }

    public void setInferModelPath(String inferModelPath) {
        inferModelPath = Common.getRealPath(inferModelPath);
        if (Common.checkPath(inferModelPath)) {
            this.inferModelPath = inferModelPath;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <inferModelPath> is not exist, please check it before set"));
            throw new RuntimeException();
        }
    }

    public String getIp() {
        if ("".equals(ip) || ip.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <ip> is null, please set it before use"));
            throw new RuntimeException();
        }
        return ip;
    }

    public void setIp(String ip) {
        if (Common.checkIP(ip)) {
            this.ip = ip;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <ip> is not valid, please check it before set"));
            throw new RuntimeException();
        }
    }

    public boolean isUseSSL() {
        return useSSL;
    }

    public void setUseSSL(boolean useSSL) {
        this.useSSL = useSSL;
    }

    public int getPort() {
        if (port == 0) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <port> is null, please set it before use"));
            throw new RuntimeException();
        }
        return port;
    }

    public void setPort(int port) {
        if (Common.checkPort(port)) {
            this.port = port;
        } else {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <port> is not valid, please check it before set"));
            throw new RuntimeException();
        }
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
        return useElb;
    }

    public void setUseElb(boolean useElb) {
        this.useElb = useElb;
    }

    public int getServerNum() {
        if (serverNum == 0) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <serverNum> is zero, please set it before use"));
            throw new RuntimeException();
        }
        return serverNum;
    }

    public void setServerNum(int serverNum) {
        this.serverNum = serverNum;
    }

    public boolean isTimer() {
        return timer;
    }

    public void setTimer(boolean timer) {
        this.timer = timer;
    }

    public int getTimeWindow() {
        return timeWindow;
    }

    public void setTimeWindow(int timeWindow) {
        this.timeWindow = timeWindow;
    }

    public int getReRequestNum() {
        return reRequestNum;
    }

    public void setReRequestNum(int reRequestNum) {
        this.reRequestNum = reRequestNum;
    }

    public String getClientID() {
        if ("".equals(clientID) || clientID.isEmpty()) {
            LOGGER.severe(Common.addTag("[flParameter] the parameter of <clientID> is null, please set it before use"));
            throw new RuntimeException();
        }
        return clientID;
    }

    public void setClientID(String clientID) {
        this.clientID = clientID;
    }

}
