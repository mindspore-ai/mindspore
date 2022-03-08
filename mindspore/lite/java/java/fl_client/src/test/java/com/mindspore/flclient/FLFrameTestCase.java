package com.mindspore.flclient;

import java.util.ArrayList;

/**
 * UT Case can be read from UT case file,
 * this class defines the content info of case using Newer frame(model code split from the frame)
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
public class FLFrameTestCase {
    private String caseName;
    // define input parameters
    private String trainDataPath;
    private String evalDataPath;
    private String inferDataPath;
    private String pathRegex;
    private String flName;
    private String trainModelPath;
    private String inferModelPath;
    private String sslProtocol;
    private String deployEnv;
    private String domainName;
    private String certPath;
    private String useElb;
    private String serverNum="1";
    private String task;
    private String threadNum="1";
    private String cpuBindMode;
    private String trainWeightName;
    private String inferWeightName;
    private String nameRegex;
    private String serverMod;
    private String inputShape;
    private String batchSize="32";
    // define http response
    private ArrayList<FLHttpRes> httpRes;
    private int resultCode;

    public String getCaseName() {
        return caseName;
    }

    public void setCaseName(String caseName) {
        this.caseName = caseName;
    }

    public String getTrainDataPath() {
        return trainDataPath;
    }

    public void setTrainDataPath(String trainDataPath) {
        this.trainDataPath = trainDataPath;
    }

    public String getEvalDataPath() {
        return evalDataPath;
    }

    public void setEvalDataPath(String evalDataPath) {
        this.evalDataPath = evalDataPath;
    }

    public String getInferDataPath() {
        return inferDataPath;
    }

    public void setInferDataPath(String inferDataPath) {
        this.inferDataPath = inferDataPath;
    }

    public String getPathRegex() {
        return pathRegex;
    }

    public void setPathRegex(String pathRegex) {
        this.pathRegex = pathRegex;
    }

    public String getFlName() {
        return flName;
    }

    public void setFlName(String flName) {
        this.flName = flName;
    }

    public String getTrainModelPath() {
        return trainModelPath;
    }

    public void setTrainModelPath(String trainModelPath) {
        this.trainModelPath = trainModelPath;
    }

    public String getInferModelPath() {
        return inferModelPath;
    }

    public void setInferModelPath(String inferModelPath) {
        this.inferModelPath = inferModelPath;
    }

    public String getSslProtocol() {
        return sslProtocol;
    }

    public void setSslProtocol(String sslProtocol) {
        this.sslProtocol = sslProtocol;
    }

    public String getDeployEnv() {
        return deployEnv;
    }

    public void setDeployEnv(String deployEnv) {
        this.deployEnv = deployEnv;
    }

    public String getDomainName() {
        return domainName;
    }

    public void setDomainName(String domainName) {
        this.domainName = domainName;
    }

    public String getCertPath() {
        return certPath;
    }

    public void setCertPath(String certPath) {
        this.certPath = certPath;
    }

    public String getUseElb() {
        return useElb;
    }

    public void setUseElb(String useElb) {
        this.useElb = useElb;
    }

    public String getServerNum() {
        return serverNum;
    }

    public void setServerNum(String serverNum) {
        this.serverNum = serverNum;
    }

    public String getTask() {
        return task;
    }

    public void setTask(String task) {
        this.task = task;
    }

    public String getThreadNum() {
        return threadNum;
    }

    public void setThreadNum(String threadNum) {
        this.threadNum = threadNum;
    }

    public String getCpuBindMode() {
        return cpuBindMode;
    }

    public void setCpuBindMode(String cpuBindMode) {
        this.cpuBindMode = cpuBindMode;
    }

    public String getTrainWeightName() {
        return trainWeightName;
    }

    public void setTrainWeightName(String trainWeightName) {
        this.trainWeightName = trainWeightName;
    }

    public String getInferWeightName() {
        return inferWeightName;
    }

    public void setInferWeightName(String inferWeightName) {
        this.inferWeightName = inferWeightName;
    }

    public String getNameRegex() {
        return nameRegex;
    }

    public void setNameRegex(String nameRegex) {
        this.nameRegex = nameRegex;
    }

    public String getServerMod() {
        return serverMod;
    }

    public void setServerMod(String serverMod) {
        this.serverMod = serverMod;
    }

    public String getInputShape() {
        return inputShape;
    }

    public void setInputShape(String inputShape) {
        this.inputShape = inputShape;
    }

    public String getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(String batchSize) {
        this.batchSize = batchSize;
    }

    public ArrayList<FLHttpRes> getHttpRes() {
        return httpRes;
    }

    public void setHttpRes(ArrayList<FLHttpRes> httpRes) {
        this.httpRes = httpRes;
    }

    public int getResultCode() {
        return resultCode;
    }

    public void setResultCode(int resultCode) {
        this.resultCode = resultCode;
    }

    void getRealPath(String basePath) {
        trainDataPath = FLUTCommon.getRealPathsWithSplit(basePath, trainDataPath, pathRegex);
        evalDataPath = FLUTCommon.getRealPathsWithSplit(basePath, evalDataPath, pathRegex);
        inferDataPath = FLUTCommon.getRealPathsWithSplit(basePath, inferDataPath, pathRegex);
        trainModelPath = FLUTCommon.getRealPathsWithSplit(basePath, trainModelPath, pathRegex);
        inferModelPath = FLUTCommon.getRealPathsWithSplit(basePath, inferModelPath, pathRegex);
        certPath = FLUTCommon.getRealPathsWithSplit(basePath, certPath, pathRegex);

        for (FLHttpRes res : httpRes) {
            if (res.getContendMode() != 0) {
                continue;
            }
            String contendFile = FLUTCommon.getRealPathsWithSplit(basePath, res.getContentData(), ",");
            res.setContentData(contendFile);
        }
    }

    String[] getParams() {
        return new String[]{
                trainDataPath,
                evalDataPath,
                inferDataPath,
                pathRegex,
                flName,
                trainModelPath,
                inferModelPath,
                sslProtocol,
                deployEnv,
                domainName,
                certPath,
                useElb,
                serverNum,
                task,
                threadNum,
                cpuBindMode,
                trainWeightName,
                inferWeightName,
                nameRegex,
                serverMod,
                batchSize,
                inputShape
        };
    }
}
