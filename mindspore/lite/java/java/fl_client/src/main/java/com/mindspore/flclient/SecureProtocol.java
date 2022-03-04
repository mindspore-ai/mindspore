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

import com.google.flatbuffers.FlatBufferBuilder;

import mindspore.schema.FeatureMap;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Map;
import java.util.List;
import java.util.HashMap;
import java.util.logging.Logger;

/**
 * Defines encryption and decryption methods.
 *
 * @since 2021-06-30
 */
public class SecureProtocol {
    private static final Logger LOGGER = Logger.getLogger(SecureProtocol.class.toString());
    private static double deltaError = 1e-6d;
    private static Map<String, float[]> modelMap;

    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private int iteration;
    private CipherClient cipherClient;
    private FLClientStatus status;
    private float[] featureMask = new float[0];
    private double dpEps;
    private double dpDelta;
    private double dpNormClip;
    private ArrayList<String> updateFeatureName = new ArrayList<String>();
    private int retCode;
    private float signK;
    private float signEps;
    private float signThrRatio;
    private float signGlobalLr;
    private int signDimOut;


    /**
     * Obtain current status code in client.
     *
     * @return current status code in client.
     */
    public FLClientStatus getStatus() {
        return status;
    }

    /**
     * Obtain retCode returned by server.
     *
     * @return the retCode returned by server.
     */
    public int getRetCode() {
        return retCode;
    }

    /**
     * Setting parameters for pairwise masking.
     *
     * @param iter         current iteration of federated learning task.
     * @param minSecretNum minimum number of secret fragments required to reconstruct a secret
     * @param prime        teh big prime number used to split secrets into pieces
     * @param featureSize  the total feature size in model
     */
    public void setPWParameter(int iter, int minSecretNum, byte[] prime, int featureSize) {
        if (prime == null || prime.length == 0) {
            LOGGER.severe(Common.addTag("[PairWiseMask] the input argument <prime> is null, please check!"));
            throw new IllegalArgumentException();
        }
        this.iteration = iter;
        this.cipherClient = new CipherClient(iteration, minSecretNum, prime, featureSize);
    }

    /**
     * Setting parameters for differential privacy.
     *
     * @param iter      current iteration of federated learning task.
     * @param diffEps   privacy budget eps of DP mechanism.
     * @param diffDelta privacy budget delta of DP mechanism.
     * @param diffNorm  normClip factor of DP mechanism.
     * @param map       model weights.
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus setDPParameter(int iter, double diffEps, double diffDelta, double diffNorm, Map<String,
            float[]> map) {
        this.iteration = iter;
        this.dpEps = diffEps;
        this.dpDelta = diffDelta;
        this.dpNormClip = diffNorm;
        this.modelMap = map;
        return FLClientStatus.SUCCESS;
    }

    /**
     * Setting parameters for dimension select.
     *
     * @param map model weights.
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus setDSParameter(float signK, float signEps, float signThrRatio, float signGlobalLr, int signDimOut, Map<String, float[]> map) {
        this.signK = signK;
        this.signEps = signEps;
        this.signThrRatio = signThrRatio;
        this.signGlobalLr = signGlobalLr;
        this.signDimOut = signDimOut;
        this.modelMap = map;
        return FLClientStatus.SUCCESS;
    }

    /**
     * Obtain the feature names that needed to be encrypted.
     *
     * @return the feature names that needed to be encrypted.
     */
    public ArrayList<String> getUpdateFeatureName() {
        return updateFeatureName;
    }

    /**
     * Set the parameter updateFeatureName.
     *
     * @param updateFeatureName the feature names that needed to be encrypted.
     */
    public void setUpdateFeatureName(ArrayList<String> updateFeatureName) {
        this.updateFeatureName = updateFeatureName;
    }

    /**
     * Obtain the returned timestamp for next request from server.
     *
     * @return the timestamp for next request.
     */
    public String getNextRequestTime() {
        return cipherClient.getNextRequestTime();
    }

    /**
     * Generate pairwise mask and individual mask.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus pwCreateMask() {
        LOGGER.info(String.format("[PairWiseMask] ==============request flID: %s ==============",
                localFLParameter.getFlID()));
        // round 0
        if (localFLParameter.isStopJobFlag()) {
            LOGGER.info(Common.addTag("the stopJObFlag is set to true, the job will be stop"));
            return status;
        }
        status = cipherClient.exchangeKeys();
        retCode = cipherClient.getRetCode();
        LOGGER.info(String.format("[PairWiseMask] ============= RequestExchangeKeys+GetExchangeKeys response: %s ",
                "============", status));
        if (status != FLClientStatus.SUCCESS) {
            return status;
        }
        // round 1
        if (localFLParameter.isStopJobFlag()) {
            LOGGER.info(Common.addTag("the stopJObFlag is set to true, the job will be stop"));
            return status;
        }
        status = cipherClient.shareSecrets();
        retCode = cipherClient.getRetCode();
        LOGGER.info(String.format("[Encrypt] =============RequestShareSecrets+GetShareSecrets response: %s ",
                "=============", status));
        if (status != FLClientStatus.SUCCESS) {
            return status;
        }
        // round2
        if (localFLParameter.isStopJobFlag()) {
            LOGGER.info(Common.addTag("the stopJObFlag is set to true, the job will be stop"));
            return status;
        }
        featureMask = cipherClient.doubleMaskingWeight();
        if (featureMask == null || featureMask.length <= 0) {
            LOGGER.severe(Common.addTag("[Encrypt] the returned featureMask from cipherClient.doubleMaskingWeight" +
                    " is null, please check!"));
            return FLClientStatus.FAILED;
        }
        retCode = cipherClient.getRetCode();
        LOGGER.info("[Encrypt] =============Create double feature mask: SUCCESS=============");
        return status;
    }

    /**
     * Add the pairwise mask and individual mask to model weights.
     *
     * @param builder       the FlatBufferBuilder object used for serialization model weights.
     * @param trainDataSize trainDataSize tne size of train data set.
     * @return the serialized model weights after adding masks.
     */
    public int[] pwMaskModel(FlatBufferBuilder builder, int trainDataSize, Map<String, float[]> trainedMap) {
        if (featureMask == null || featureMask.length == 0) {
            LOGGER.severe("[Encrypt] feature mask is null, please check");
            return new int[0];
        }
        LOGGER.info(String.format("[Encrypt] feature mask size: %s", featureMask.length));
        int featureSize = updateFeatureName.size();
        int[] featuresMap = new int[featureSize];
        int maskIndex = 0;
        for (int i = 0; i < featureSize; i++) {
            String key = updateFeatureName.get(i);
            float[] data = trainedMap.get(key);
            LOGGER.info(String.format("[Encrypt] feature name: %s feature size: %s", key, data.length));
            for (int j = 0; j < data.length; j++) {
                float rawData = data[j];
                if (maskIndex >= featureMask.length) {
                    LOGGER.severe("[Encrypt] the maskIndex is out of range for array featureMask, please check");
                    return new int[0];
                }
                float maskData = rawData * trainDataSize + featureMask[maskIndex];
                maskIndex += 1;
                data[j] = maskData;
            }
            int featureName = builder.createString(key);
            int weight = FeatureMap.createDataVector(builder, data);
            int featureMap = FeatureMap.createFeatureMap(builder, featureName, weight);
            featuresMap[i] = featureMap;
        }
        return featuresMap;
    }

    /**
     * Reconstruct the secrets used for unmasking model weights.
     *
     * @return current status code in client.
     */
    public FLClientStatus pwUnmasking() {
        status = cipherClient.reconstructSecrets();   // round3
        retCode = cipherClient.getRetCode();
        LOGGER.info(String.format("[Encrypt] =============GetClientList+SendReconstructSecret: %s =============",
                status));
        return status;
    }

    private static float calculateErf(double erfInput) {
        double result = 0d;
        int segmentNum = 10000;
        double deltaX = erfInput / segmentNum;
        result += 1;
        for (int i = 1; i < segmentNum; i++) {
            result += 2 * Math.exp(-Math.pow(deltaX * i, 2));
        }
        result += Math.exp(-Math.pow(deltaX * segmentNum, 2));
        return (float) (result * deltaX / Math.pow(Math.PI, 0.5));
    }

    private static double calculatePhi(double phiInput) {
        return 0.5 * (1.0 + calculateErf((phiInput / Math.sqrt(2.0))));
    }

    private static double calculateBPositive(double eps, double calInput) {
        return calculatePhi(Math.sqrt(eps * calInput)) -
                Math.exp(eps) * calculatePhi(-Math.sqrt(eps * (calInput + 2.0)));
    }

    private static double calculateBNegative(double eps, double calInput) {
        return calculatePhi(-Math.sqrt(eps * calInput)) -
                Math.exp(eps) * calculatePhi(-Math.sqrt(eps * (calInput + 2.0)));
    }

    private static double calculateSPositive(double eps, double targetDelta, double initSInf, double initSSup) {
        double deltaSup = calculateBPositive(eps, initSSup);
        double sInf = initSInf;
        double sSup = initSSup;
        while (deltaSup <= targetDelta) {
            sInf = sSup;
            sSup = 2 * sInf;
            deltaSup = calculateBPositive(eps, sSup);
        }
        double sMid = sInf + (sSup - sInf) / 2.0;
        int iterMax = 1000;
        int iters = 0;
        while (true) {
            double bPositive = calculateBPositive(eps, sMid);
            if (bPositive <= targetDelta) {
                if (targetDelta - bPositive <= deltaError) {
                    break;
                } else {
                    sInf = sMid;
                }
            } else {
                sSup = sMid;
            }
            sMid = sInf + (sSup - sInf) / 2.0;
            iters += 1;
            if (iters > iterMax) {
                break;
            }
        }
        return sMid;
    }

    private static double calculateSNegative(double eps, double targetDelta, double initSInf, double initSSup) {
        double deltaSup = calculateBNegative(eps, initSSup);
        double sInf = initSInf;
        double sSup = initSSup;
        while (deltaSup > targetDelta) {
            sInf = sSup;
            sSup = 2 * sInf;
            deltaSup = calculateBNegative(eps, sSup);
        }

        double sMid = sInf + (sSup - sInf) / 2.0;
        int iterMax = 1000;
        int iters = 0;
        while (true) {
            double bNegative = calculateBNegative(eps, sMid);
            if (bNegative <= targetDelta) {
                if (targetDelta - bNegative <= deltaError) {
                    break;
                } else {
                    sSup = sMid;
                }
            } else {
                sInf = sMid;
            }
            sMid = sInf + (sSup - sInf) / 2.0;
            iters += 1;
            if (iters > iterMax) {
                break;
            }
        }
        return sMid;
    }

    private static double calculateSigma(double clipNorm, double eps, double targetDelta) {
        double deltaZero = calculateBPositive(eps, 0);
        double alpha = 1d;
        if (targetDelta > deltaZero) {
            double sPositive = calculateSPositive(eps, targetDelta, 0, 1);
            alpha = Math.sqrt(1.0 + sPositive / 2.0) - Math.sqrt(sPositive / 2.0);
        } else if (targetDelta < deltaZero) {
            double sNegative = calculateSNegative(eps, targetDelta, 0, 1);
            alpha = Math.sqrt(1.0 + sNegative / 2.0) + Math.sqrt(sNegative / 2.0);
        } else {
            LOGGER.info(Common.addTag("[Encrypt] targetDelta = deltaZero"));
        }
        return alpha * clipNorm / Math.sqrt(2.0 * eps);
    }

    /**
     * Add differential privacy mask to model weights.
     *
     * @param builder       the FlatBufferBuilder object used for serialization model weights.
     * @param trainDataSize tne size of train data set.
     * @return the serialized model weights after adding masks.
     */
    public int[] dpMaskModel(FlatBufferBuilder builder, int trainDataSize, Map<String, float[]> trainedMap) {
        // get feature map
        Map<String, float[]> mapBeforeTrain = modelMap;
        int featureSize = updateFeatureName.size();
        // calculate sigma
        double gaussianSigma = calculateSigma(dpNormClip, dpEps, dpDelta);
        LOGGER.info(Common.addTag("[Encrypt] =============Noise sigma of DP is: " + gaussianSigma + "============="));

        // calculate l2-norm of all layers' update array
        double updateL2Norm = 0d;
        for (int i = 0; i < featureSize; i++) {
            String key = updateFeatureName.get(i);
            float[] data = trainedMap.get(key);
            float[] dataBeforeTrain = mapBeforeTrain.get(key);
            for (int j = 0; j < data.length; j++) {
                float rawData = data[j];
                if (j >= dataBeforeTrain.length) {
                    LOGGER.severe("[Encrypt] the index j is out of range for array dataBeforeTrain, please check");
                    return new int[0];
                }
                float rawDataBeforeTrain = dataBeforeTrain[j];
                float updateData = rawData - rawDataBeforeTrain;
                updateL2Norm += updateData * updateData;
            }
        }
        updateL2Norm = Math.sqrt(updateL2Norm);
        if (updateL2Norm == 0) {
            LOGGER.severe(Common.addTag("[Encrypt] updateL2Norm is 0, please check"));
            return new int[0];
        }
        double clipFactor = Math.min(1.0, dpNormClip / updateL2Norm);

        // clip and add noise
        int[] featuresMap = new int[featureSize];
        for (int i = 0; i < featureSize; i++) {
            String key = updateFeatureName.get(i);
            if (!trainedMap.containsKey(key)) {
                LOGGER.severe("[Encrypt] the key: " + key + " is not in map, please check!");
                return new int[0];
            }
            float[] data = trainedMap.get(key);
            float[] data2 = new float[data.length];
            if (!mapBeforeTrain.containsKey(key)) {
                LOGGER.severe("[Encrypt] the key: " + key + " is not in mapBeforeTrain, please check!");
                return new int[0];
            }
            float[] dataBeforeTrain = mapBeforeTrain.get(key);

            // prepare gaussian noise
            SecureRandom secureRandom = Common.getSecureRandom();
            for (int j = 0; j < data.length; j++) {
                float rawData = data[j];
                if (j >= dataBeforeTrain.length) {
                    LOGGER.severe("[Encrypt] the index j is out of range for array dataBeforeTrain, please check");
                    return new int[0];
                }
                float rawDataBeforeTrain = dataBeforeTrain[j];
                float updateData = rawData - rawDataBeforeTrain;

                // clip
                updateData *= clipFactor;

                // add noise
                double gaussianNoise = secureRandom.nextGaussian() * gaussianSigma;
                updateData += gaussianNoise;
                data2[j] = rawDataBeforeTrain + updateData;
                data2[j] = data2[j] * trainDataSize;
            }
            int featureName = builder.createString(key);
            int weight = FeatureMap.createDataVector(builder, data2);
            int featureMap = FeatureMap.createFeatureMap(builder, featureName, weight);
            featuresMap[i] = featureMap;
        }
        return featuresMap;
    }

    /**
     * The number of combinations of n things taken k.
     *
     * @param n Number of things.
     * @param k Number of elements taken.
     * @return the total number of "n choose k" combinations.
     */
    private static double comb(double n, double k) {
        boolean cond = (k <= n) && (n >= 0) && (k >= 0);
        double m = n + 1;
        if (!cond) {
            return 0;
        } else {
            double nTerm = Math.min(k, n - k);
            double res = 1;
            for (int i = 1; i <= nTerm; i++) {
                res *= (m - i);
                res /= i;
            }
            return res;
        }
    }

    /**
     * Calculate the number of possible combinations of output set given the number of topk dimensions.
     * c(k, v) * c(d-k, h-v)
     *
     * @param numInter  the number of dimensions from topk set.
     * @param topkDim   the size of top-k set.
     * @param inputDim  total number of dimensions in the model.
     * @param outputDim the number of dimensions selected for constructing sparse local updates.
     * @return the number of possible combinations of output set.
     */
    private static double countCombs(int numInter, int topkDim, int inputDim, int outputDim) {
        return comb(topkDim, numInter) * comb(inputDim - topkDim, outputDim - numInter);
    }

    /**
     * Calculate the probability mass function of the number of topk dimensions in the output set.
     * v is the number of dimensions from topk set.
     *
     * @param thr       threshold of the number of topk dimensions in the output set.
     * @param topkDim   the size of top-k set.
     * @param inputDim  total number of dimensions in the model.
     * @param outputDim the number of dimensions selected for constructing sparse local updates.
     * @param eps       the privacy budget of SignDS alg.
     * @return the probability mass function.
     */
    private static List<Double> calcPmf(int thr, int topkDim, int inputDim, int outputDim, float eps) {
        List<Double> pmf = new ArrayList<>();
        double newPmf;
        for (int v = 0; v <= outputDim; v++) {
            if (v < thr) {
                newPmf = countCombs(v, topkDim, inputDim, outputDim);
            } else {
                newPmf = countCombs(v, topkDim, inputDim, outputDim) * Math.exp(eps);
            }
            pmf.add(newPmf);
        }
        double pmfSum = 0;
        for (int i = 0; i < pmf.size(); i++) {
            pmfSum += pmf.get(i);
        }
        if (pmfSum == 0) {
            LOGGER.severe(Common.addTag("[SignDS] probability mass function is 0, please check"));
            return new ArrayList<>();
        }
        for (int i = 0; i < pmf.size(); i++) {
            pmf.set(i, pmf.get(i) / pmfSum);
        }
        return pmf;
    }

    /**
     * Calculate the expected number of topk dimensions in the output set given outputDim.
     * The size of pmf is also outputDim.
     *
     * @param pmf probability mass function
     * @return the expectation of the topk dimensions in the output set.
     */
    private static double calcExpectation(List<Double> pmf) {
        double sumExpectation = 0;
        for (int i = 0; i < pmf.size(); i++) {
            sumExpectation += (i * pmf.get(i));
        }
        return sumExpectation;
    }

    /**
     * Calculate the optimum threshold for the number of topk dimension in the output set.
     * The optimum threshold is an integer among [1, outputDim], which has the largest
     * expectation value.
     *
     * @param topkDim   the size of top-k set.
     * @param inputDim  total number of dimensions in the model.
     * @param outputDim the number of dimensions selected for constructing sparse local updates.
     * @param eps       the privacy budget of SignDS alg.
     * @return the optimum threshold.
     */
    private static int calcOptThr(int topkDim, int inputDim, int outputDim, float eps) {
        double optExpect = 0;
        double optT = 0;
        for (int t = 1; t <= outputDim; t++) {
            double newExpect = calcExpectation(calcPmf(t, topkDim, inputDim, outputDim, eps));
            if (newExpect > optExpect) {
                optExpect = newExpect;
                optT = t;
            } else {
                break;
            }
        }
        return (int) Math.max(optT, 1);
    }

    /**
     * Tool function for finding the optimum output dimension.
     * The main idea is to iteratively search for the largest output dimension while
     * ensuring the expected ratio of topk dimensions in the output set larger than
     * the target ratio.
     *
     * @param thrInterRatio threshold of the expected ratio of topk dimensions
     * @param topkDim       the size of top-k set.
     * @param inputDim      total number of dimensions in the model.
     * @param eps           the privacy budget of SignDS alg.
     * @return the optimum output dimension.
     */
    private static int findOptOutputDim(float thrInterRatio, int topkDim, int inputDim, float eps) {
        int outputDim = 1;
        while (true) {
            int thr = calcOptThr(topkDim, inputDim, outputDim, eps);
            double expectedRatio = calcExpectation(calcPmf(thr, topkDim, inputDim, outputDim, eps)) / outputDim;
            if (expectedRatio < thrInterRatio || Double.isNaN(expectedRatio)) {
                break;
            } else {
                outputDim += 1;
            }
        }
        return Math.max(1, (outputDim - 1));
    }

    /**
     * Determine the number of dimensions to be sampled from the topk dimension set via
     * inverse sampling.
     * The main steps of the trick of inverse sampling include:
     * 1. Sample a random probability from the uniform distribution U(0, 1).
     * 2. Calculate the cumulative distribution of numInter, namely the number of
     * topk dimensions in the output set.
     * 3. Compare the cumulative distribution with the random probability and determine
     * the value of numInter.
     *
     * @param thrDim      threshold of the number of topk dimensions in the output set.
     * @param denominator calculate denominator given the threshold.
     * @param topkDim     the size of top-k set.
     * @param inputDim    total number of dimensions in the model.
     * @param outputDim   the number of dimensions selected for constructing sparse local updates.
     * @param eps         the privacy budget of SignDS alg.
     * @return the number of dimensions to be sampled from the top-k dimension set.
     */
    private static int countInters(int thrDim, double denominator, int topkDim, int inputDim, int outputDim, float eps) {
        SecureRandom secureRandom = new SecureRandom();
        double randomProb = secureRandom.nextDouble();
        int numInter = 0;
        double prob = countCombs(numInter, topkDim, inputDim, outputDim) / denominator;
        while (prob < randomProb) {
            numInter += 1;
            if (numInter < thrDim) {
                prob += countCombs(numInter, topkDim, inputDim, outputDim) / denominator;
            } else {
                prob += Math.exp(eps) * countCombs(numInter, topkDim, inputDim, outputDim) / denominator;
            }
        }
        return numInter;
    }

    /**
     * select num indexes from inputList, and put them into outputList.
     *
     * @param secureRandom cryptographically strong random number generator.
     * @param inputList    select index from inputList.
     * @param outputList   put random index into outputList.
     * @param num          the number of select indexes.
     */
    private static void randomSelect(SecureRandom secureRandom, List<Integer> inputList, List<Integer> outputList, int num) {
        if (num <= 0) {
            LOGGER.severe(Common.addTag("[SignDS] The number to be selected is set incorrectly!"));
            return;
        }
        if (inputList.isEmpty()) {
            LOGGER.severe(Common.addTag("[SignDS] The input List is empty!"));
            return;
        }
        if (inputList.size() < num) {
            LOGGER.severe(Common.addTag("[SignDS] The size of inputList is small than num!"));
            return;
        }
        for (int i = inputList.size(); i > inputList.size() - num; i--) {
            int randomIndex = secureRandom.nextInt(i);
            int randomSelectTopkIndex = inputList.get(randomIndex);
            inputList.set(randomIndex, inputList.get(i - 1));
            inputList.set(i - 1, randomSelectTopkIndex);
            outputList.add(randomSelectTopkIndex);
        }
    }

    /**
     * SignDS alg.
     *
     * @param trainedMap trained model.
     * @param sign       random sign value.
     * @return index list.
     */
    public int[] signDSModel(Map<String, float[]> trainedMap, boolean sign) {
        Map<String, float[]> mapBeforeTrain = modelMap;
        int layerNum = updateFeatureName.size();
        SecureRandom secureRandom = Common.getSecureRandom();
        List<Integer> nonTopkKeyList = new ArrayList<>();
        List<Integer> topkKeyList = new ArrayList<>();
        Map<Integer, Float> allUpdateMap = new HashMap<>();
        int index = 0;
        for (int i = 0; i < layerNum; i++) {
            String key = updateFeatureName.get(i);
            float[] dataAfterTrain = trainedMap.get(key);
            float[] dataBeforeTrain = mapBeforeTrain.get(key);
            for (int j = 0; j < dataAfterTrain.length; j++) {
                float updateData = dataAfterTrain[j] - dataBeforeTrain[j];
                allUpdateMap.put(index++, updateData);
            }
        }
        int inputDim = allUpdateMap.size();
        int topkDim = (int) (signK * inputDim);
        if (signDimOut == 0) {
            signDimOut = findOptOutputDim(signThrRatio, topkDim, inputDim, signEps);
        }
        int thrDim = calcOptThr(topkDim, inputDim, signDimOut, signEps);
        double combLessInter = 0d;
        double combMoreInter = 0d;
        for (int i = 0; i < thrDim; i++) {
            combLessInter += countCombs(i, topkDim, inputDim, signDimOut);
        }
        for (int i = thrDim; i <= signDimOut; i++) {
            combMoreInter += countCombs(i, topkDim, inputDim, signDimOut);
        }
        double denominator = combLessInter + Math.exp(signEps) * combMoreInter;
        if (denominator == 0) {
            LOGGER.severe(Common.addTag("[SignDS] denominator is 0, please check"));
            return new int[0];
        }
        int numInter = countInters(thrDim, denominator, topkDim, inputDim, signDimOut, signEps);
        int numOuter = signDimOut - numInter;
        if (topkDim < numInter || signDimOut <= 0) {
            LOGGER.severe("[SignDS] topkDim or signDimOut is ERROR! please check");
            return new int[0];
        }
        List<Map.Entry<Integer, Float>> allUpdateList = new ArrayList<>(allUpdateMap.entrySet());
        if (sign) {
            allUpdateList.sort((o1, o2) -> Float.compare(o2.getValue(), o1.getValue()));
        } else {
            allUpdateList.sort((o1, o2) -> Float.compare(o1.getValue(), o2.getValue()));
        }
        for (int i = 0; i < topkDim; i++) {
            topkKeyList.add(allUpdateList.get(i).getKey());
        }
        for (int i = topkDim; i < allUpdateList.size(); i++) {
            nonTopkKeyList.add(allUpdateList.get(i).getKey());
        }
        List<Integer> outputDimensionIndexList = new ArrayList<>();
        randomSelect(secureRandom, topkKeyList, outputDimensionIndexList, numInter);
        randomSelect(secureRandom, nonTopkKeyList, outputDimensionIndexList, numOuter);
        outputDimensionIndexList.sort(Integer::compare);
        LOGGER.info(Common.addTag("[SignDS] outputDimension size is " + outputDimensionIndexList.size()));
        return outputDimensionIndexList.stream().mapToInt(i -> i).toArray();
    }
}
