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
import static com.mindspore.flclient.LocalFLParameter.LENET;

import com.google.flatbuffers.FlatBufferBuilder;

import com.mindspore.flclient.model.AlTrainBert;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.TrainLenet;

import mindspore.schema.FeatureMap;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
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
    private ArrayList<String> encryptFeatureName = new ArrayList<String>();
    private int retCode;

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
     * Obtain the feature names that needed to be encrypted.
     *
     * @return the feature names that needed to be encrypted.
     */
    public ArrayList<String> getEncryptFeatureName() {
        return encryptFeatureName;
    }

    /**
     * Set the parameter encryptFeatureName.
     *
     * @param encryptFeatureName the feature names that needed to be encrypted.
     */
    public void setEncryptFeatureName(ArrayList<String> encryptFeatureName) {
        this.encryptFeatureName = encryptFeatureName;
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
        status = cipherClient.exchangeKeys();
        retCode = cipherClient.getRetCode();
        LOGGER.info(String.format("[PairWiseMask] ============= RequestExchangeKeys+GetExchangeKeys response: %s ",
                "============", status));
        if (status != FLClientStatus.SUCCESS) {
            return status;
        }
        // round 1
        status = cipherClient.shareSecrets();
        retCode = cipherClient.getRetCode();
        LOGGER.info(String.format("[Encrypt] =============RequestShareSecrets+GetShareSecrets response: %s ",
                "=============", status));
        if (status != FLClientStatus.SUCCESS) {
            return status;
        }
        // round2
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
    public int[] pwMaskModel(FlatBufferBuilder builder, int trainDataSize) {
        if (featureMask == null || featureMask.length == 0) {
            LOGGER.severe("[Encrypt] feature mask is null, please check");
            return new int[0];
        }
        LOGGER.info(String.format("[Encrypt] feature mask size: %s", featureMask.length));
        // get feature map
        Map<String, float[]> map = new HashMap<String, float[]>();
        if (flParameter.getFlName().equals(ALBERT)) {
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(alTrainBert.getTrainSession()));
        } else if (flParameter.getFlName().equals(LENET)) {
            TrainLenet trainLenet = TrainLenet.getInstance();
            map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(trainLenet.getTrainSession()));
        } else {
            LOGGER.severe(Common.addTag("[Encrypt] the flName is not valid, only support: lenet, albert"));
            throw new IllegalArgumentException();
        }
        int featureSize = encryptFeatureName.size();
        int[] featuresMap = new int[featureSize];
        int maskIndex = 0;
        for (int i = 0; i < featureSize; i++) {
            String key = encryptFeatureName.get(i);
            float[] data = map.get(key);
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
    public int[] dpMaskModel(FlatBufferBuilder builder, int trainDataSize) {
        // get feature map
        Map<String, float[]> map = new HashMap<String, float[]>();
        if (flParameter.getFlName().equals(ALBERT)) {
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(alTrainBert.getTrainSession()));
        } else if (flParameter.getFlName().equals(LENET)) {
            TrainLenet trainLenet = TrainLenet.getInstance();
            map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(trainLenet.getTrainSession()));
        } else {
            LOGGER.severe(Common.addTag("[Encrypt] the flName is not valid, only support: lenet, albert"));
            throw new IllegalArgumentException();
        }
        Map<String, float[]> mapBeforeTrain = modelMap;
        int featureSize = encryptFeatureName.size();
        // calculate sigma
        double gaussianSigma = calculateSigma(dpNormClip, dpEps, dpDelta);
        LOGGER.info(Common.addTag("[Encrypt] =============Noise sigma of DP is: " + gaussianSigma + "============="));

        // calculate l2-norm of all layers' update array
        double updateL2Norm = 0d;
        for (int i = 0; i < featureSize; i++) {
            String key = encryptFeatureName.get(i);
            float[] data = map.get(key);
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
        double clipFactor = Math.min(1.0, dpNormClip / updateL2Norm);

        // clip and add noise
        int[] featuresMap = new int[featureSize];
        for (int i = 0; i < featureSize; i++) {
            String key = encryptFeatureName.get(i);
            if (!map.containsKey(key)) {
                LOGGER.severe("[Encrypt] the key: " + key + " is not in map, please check!");
                return new int[0];
            }
            float[] data = map.get(key);
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
}
