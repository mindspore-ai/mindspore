package com.mindspore.flclient;

import java.util.ArrayList;

/**
 * UT Case can be read from UT case file,
 * this class defines the content info of case using older frame(model code in the frame)
 *
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
public class FLFrameWithModelTestCase {
    private String caseName;
    // define input parameters
    private String trainDataset;
    private String vocabFile;
    private String idsFile;
    private String testDataset;
    private String flName;
    private String trainModelPath;
    private String inferModelPath;
    private String useSSL;
    private String domainName;
    private String useElb;
    private String serverNum;
    private String certPath;
    private String task;
    // define http response
    private ArrayList<FLHttpRes> httpRes;
    private int resultCode;

    // define the result
    // TODO:how to check?

    public String getCaseName() {
        return caseName;
    }

    public void setCaseName(String caseName) {
        this.caseName = caseName;
    }

    public String getTrainDataset() {
        return trainDataset;
    }

    public void setTrainDataset(String trainDataset) {
        this.trainDataset = trainDataset;
    }

    public String getVocabFile() {
        return vocabFile;
    }

    public void setVocabFile(String vocabFile) {
        this.vocabFile = vocabFile;
    }

    public String getIdsFile() {
        return idsFile;
    }

    public void setIdsFile(String idsFile) {
        this.idsFile = idsFile;
    }

    public String getTestDataset() {
        return testDataset;
    }

    public void setTestDataset(String testDataset) {
        this.testDataset = testDataset;
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

    public String getUseSSL() {
        return useSSL;
    }

    public void setUseSSL(String useSSL) {
        this.useSSL = useSSL;
    }

    public String getDomainName() {
        return domainName;
    }

    public void setDomainName(String domainName) {
        this.domainName = domainName;
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

    public String getCertPath() {
        return certPath;
    }

    public void setCertPath(String certPath) {
        this.certPath = certPath;
    }

    public String getTask() {
        return task;
    }

    public void setTask(String task) {
        this.task = task;
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


    String[] getParams() {
        return new String[]{trainDataset,
                vocabFile,
                idsFile,
                testDataset,
                flName,
                trainModelPath,
                inferModelPath,
                useSSL,
                domainName,
                useElb,
                serverNum,
                certPath,
                task};
    }

    void getRealPath(String basePath) {
        trainDataset = FLUTCommon.getRealPathsWithSplit(basePath, trainDataset, ",");
        vocabFile = FLUTCommon.getRealPathsWithSplit(basePath, vocabFile, ",");
        idsFile = FLUTCommon.getRealPathsWithSplit(basePath, idsFile, ",");
        testDataset = FLUTCommon.getRealPathsWithSplit(basePath, testDataset, ",");
        trainModelPath = FLUTCommon.getRealPathsWithSplit(basePath, trainModelPath, ",");
        inferModelPath = FLUTCommon.getRealPathsWithSplit(basePath, inferModelPath, ",");
        certPath = FLUTCommon.getRealPathsWithSplit(basePath, certPath, ",");

        for (FLHttpRes res : httpRes) {
            if (res.getContendMode() != 0) {
                continue;
            }
            String contendFile = FLUTCommon.getRealPathsWithSplit(basePath, res.getContentData(), ",");
            res.setContentData(contendFile);
        }
    }
}
