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
import com.mindspore.flclient.model.AlTrainBert;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.TrainLenet;
import mindspore.schema.FeatureMap;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.logging.Logger;

public class SecureProtocol {
    private static final Logger LOGGER = Logger.getLogger(SecureProtocol.class.toString());
    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private int iteration;
    private CipherClient cipher;
    private FLClientStatus status;
    private float[] featureMask;
    private double dpEps;
    private double dpDelta;
    private double dpNormClip;
    private static double deltaError = 1e-6;
    private static Map<String, float[]> modelMap;
    private ArrayList<String> encryptFeatureName = new ArrayList<String>();
    private int retCode;

    public FLClientStatus getStatus() {
        return status;
    }

    public float[] getFeatureMask() {
        return featureMask;
    }

    public int getRetCode() {
        return retCode;
    }

    public SecureProtocol() {
    }

    public void setPWParameter(int iter, int minSecretNum, byte[] prime, int featureSize) {
        this.iteration = iter;
        this.cipher = new CipherClient(iteration, minSecretNum, prime, featureSize);
    }

    public FLClientStatus setDPParameter(int iter, double diffEps,
                                         double diffDelta, double diffNorm, Map<String, float[]> map) {
        try {
            this.iteration = iter;
            this.dpEps = diffEps;
            this.dpDelta = diffDelta;
            this.dpNormClip = diffNorm;
            this.modelMap = map;
            status = FLClientStatus.SUCCESS;
        } catch (Exception e) {
            LOGGER.severe(Common.addTag("[DPEncrypt] catch Exception in setDPParameter: " + e.getMessage()));
            status = FLClientStatus.FAILED;
        }
        return status;
    }

    public ArrayList<String> getEncryptFeatureName() {
        return encryptFeatureName;
    }

    public void setEncryptFeatureName(ArrayList<String> encryptFeatureName) {
        this.encryptFeatureName = encryptFeatureName;
    }

    public String getNextRequestTime() {
        return cipher.getNextRequestTime();
    }

    public FLClientStatus pwCreateMask() {
        LOGGER.info("[PairWiseMask] ==============request flID: " + localFLParameter.getFlID() + "==============");
        // round 0
        status = cipher.exchangeKeys();
        retCode = cipher.getRetCode();
        LOGGER.info("[PairWiseMask] ============= RequestExchangeKeys+GetExchangeKeys response: " + status + "============");
        if (status != FLClientStatus.SUCCESS) {
            return status;
        }
        // round 1
        try {
            status = cipher.shareSecrets();
            retCode = cipher.getRetCode();
            LOGGER.info("[Encrypt] =============RequestShareSecrets+GetShareSecrets response: " + status + "=============");
        } catch (Exception e) {
            LOGGER.severe("[PairWiseMask] catch Exception in pwCreateMask");
            status = FLClientStatus.FAILED;
        }
        if (status != FLClientStatus.SUCCESS) {
            return status;
        }
        // round2
        try {
            featureMask = cipher.doubleMaskingWeight();
            retCode = cipher.getRetCode();
            LOGGER.info("[Encrypt] =============Create double feature mask: SUCCESS=============");
        } catch (Exception e) {
            LOGGER.severe("[PairWiseMask] catch Exception in pwCreateMask");
            status = FLClientStatus.FAILED;
        }
        return status;
    }

    public int[] pwMaskModel(FlatBufferBuilder builder, int trainDataSize) {
        LOGGER.info("[Encrypt] feature mask size: " + featureMask.length);
        // get feature map
        Map<String, float[]> map = new HashMap<String, float[]>();
        if (flParameter.getFlName().equals("adbert")) {
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(alTrainBert.getTrainSession()));
        } else if (flParameter.getFlName().equals("lenet")) {
            TrainLenet trainLenet = TrainLenet.getInstance();
            map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(trainLenet.getTrainSession()));
        }
        int featureSize = encryptFeatureName.size();
        int[] featuresMap = new int[featureSize];
        int maskIndex = 0;
        for (int i = 0; i < featureSize; i++) {
            String key = encryptFeatureName.get(i);
            float[] data = map.get(key);
            LOGGER.info("[Encrypt] feature name: " + key + " feature size: " + data.length);
            for (int j = 0; j < data.length; j++) {
                float rawData = data[j];
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

    public FLClientStatus pwUnmasking() {
        status = cipher.reconstructSecrets();   // round3
        retCode = cipher.getRetCode();
        LOGGER.info("[Encrypt] =============GetClientList+SendReconstructSecret: " + status + "=============");
        return status;
    }

    private static float calculateErf(double x) {
        double result = 0;
        int segmentNum = 10000;
        double deltaX = x / segmentNum;
        result += 1;
        for (int i = 1; i < segmentNum; i++) {
            result += 2 * Math.exp(-Math.pow(deltaX * i, 2));
        }
        result += Math.exp(-Math.pow(deltaX * segmentNum, 2));
        return (float) (result * deltaX / Math.pow(Math.PI, 0.5));
    }

    private static double calculatePhi(double t) {
        return 0.5 * (1.0 + calculateErf((t / Math.sqrt(2.0))));
    }

    private static double calculateBPositive(double eps, double s) {
        return calculatePhi(Math.sqrt(eps * s)) - Math.exp(eps) * calculatePhi(-Math.sqrt(eps * (s + 2.0)));
    }

    private static double calculateBNegative(double eps, double s) {
        return calculatePhi(-Math.sqrt(eps * s)) - Math.exp(eps) * calculatePhi(-Math.sqrt(eps * (s + 2.0)));
    }

    private static double calculateSPositive(double eps, double targetDelta, double sInf, double sSup) {
        double deltaSup = calculateBPositive(eps, sSup);
        while (deltaSup <= targetDelta) {
            sInf = sSup;
            sSup = 2 * sInf;
            deltaSup = calculateBPositive(eps, sSup);
        }

        double sMid = sInf + (sSup - sInf) / 2.0;
        int iterMax = 1000;
        int iters = 0;
        while (true) {
            double b = calculateBPositive(eps, sMid);
            if (b <= targetDelta) {
                if (targetDelta - b <= deltaError) {
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

    private static double calculateSNegative(double eps, double targetDelta, double sInf, double sSup) {
        double deltaSup = calculateBNegative(eps, sSup);
        while (deltaSup > targetDelta) {
            sInf = sSup;
            sSup = 2 * sInf;
            deltaSup = calculateBNegative(eps, sSup);
        }

        double sMid = sInf + (sSup - sInf) / 2.0;
        int iterMax = 1000;
        int iters = 0;
        while (true) {
            double b = calculateBNegative(eps, sMid);
            if (b <= targetDelta) {
                if (targetDelta - b <= deltaError) {
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
        double alpha = 1;
        if (targetDelta > deltaZero) {
            double s = calculateSPositive(eps, targetDelta, 0, 1);
            alpha = Math.sqrt(1.0 + s / 2.0) - Math.sqrt(s / 2.0);
        } else if (targetDelta < deltaZero) {
            double s = calculateSNegative(eps, targetDelta, 0, 1);
            alpha = Math.sqrt(1.0 + s / 2.0) + Math.sqrt(s / 2.0);
        }
        return alpha * clipNorm / Math.sqrt(2.0 * eps);
    }

    public int[] dpMaskModel(FlatBufferBuilder builder, int trainDataSize) {
        // get feature map
        Map<String, float[]> map = new HashMap<String, float[]>();
        if (flParameter.getFlName().equals("adbert")) {
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(alTrainBert.getTrainSession()));
        } else if (flParameter.getFlName().equals("lenet")) {
            TrainLenet trainLenet = TrainLenet.getInstance();
            map = SessionUtil.convertTensorToFeatures(SessionUtil.getFeatures(trainLenet.getTrainSession()));
        }
        Map<String, float[]> mapBeforeTrain = modelMap;
        int featureSize = encryptFeatureName.size();
        // calculate sigma
        double gaussianSigma = calculateSigma(dpNormClip, dpEps, dpDelta);
        LOGGER.info(Common.addTag("[Encrypt] =============Noise sigma of DP is: " + gaussianSigma + "============="));

        // prepare gaussian noise
        SecureRandom random = new SecureRandom();
        int randomInt = random.nextInt();
        Random r = new Random(randomInt);

        // calculate l2-norm of all layers' update array
        double updateL2Norm = 0;
        for (int i = 0; i < featureSize; i++) {
            String key = encryptFeatureName.get(i);
            float[] data = map.get(key);
            float[] dataBeforeTrain = mapBeforeTrain.get(key);
            for (int j = 0; j < data.length; j++) {
                float rawData = data[j];
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
            float[] data = map.get(key);
            float[] data2 = new float[data.length];
            float[] dataBeforeTrain = mapBeforeTrain.get(key);
            for (int j = 0; j < data.length; j++) {
                float rawData = data[j];
                float rawDataBeforeTrain = dataBeforeTrain[j];
                float updateData = rawData - rawDataBeforeTrain;

                // clip
                updateData *= clipFactor;

                // add noise
                double gaussianNoise = r.nextGaussian() * gaussianSigma;
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
