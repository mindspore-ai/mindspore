/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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

package com.mindspore.flclient.compression;

import com.mindspore.flclient.LocalFLParameter;
import static mindspore.schema.CompressType.DIFF_SPARSE_QUANT;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.PriorityQueue;

/**
 * Encode Executor
 *
 * @since 2021-12-21
 */
public class EncodeExecutor {
    private final LocalFLParameter localFLParameter = LocalFLParameter.getInstance();

    private static volatile EncodeExecutor encodeExecutor;

    private EncodeExecutor() {}

    public static EncodeExecutor getInstance() {
        if (encodeExecutor == null) {
            synchronized (EncodeExecutor.class) {
                if (encodeExecutor == null) {
                    encodeExecutor = new EncodeExecutor();
                }
            }
        }
        return encodeExecutor;
    }

    private static final int multiplier = 2147483647;
    private static final double increment = 4294967294.0;
    private static final int modulo = 48271;

    private List<Integer> constructMaskArray(int paramNum) {
        int seed = localFLParameter.getSeed();
        float uploadSparseRatio = localFLParameter.getUploadSparseRatio();

        List<Integer> maskArray = new ArrayList<>();

        int retain_num = (int) ((float) (paramNum) * uploadSparseRatio);
        for (int i = 0; i < retain_num; ++i) {
            maskArray.add(1);
        }
        for (int i = retain_num; i < paramNum; ++i) {
            maskArray.add(0);
        }

        seed = ((seed + multiplier) * modulo) % multiplier;
        for (int i = 0; i < paramNum; ++i) {
            // generate random number in (0, 1)
            double rand = (double)(seed) / increment + 0.5;
            // update seed
            seed = (seed * modulo) % multiplier;

            int j = (int)(rand * (double)(paramNum - i)) + i;
            int temp = maskArray.get(i);
            maskArray.set(i, maskArray.get(j));
            maskArray.set(j, temp);
        }
        return maskArray;
    }

    public List<CompressWeight> enDiffSparseQuant(Map<String, List<Float>> featureMaps, int numBits,
                                                  int trainDataSize) {
        List<CompressWeight> compressWeights = new ArrayList<>();

        // difference encode
        Map<String, float[]> oldFeatureMap = localFLParameter.getOldFeatureMap();
        Map<String, List<Float>> diffFeatureMaps = new HashMap<>();
        for (String featureMapName : featureMaps.keySet()) {
            List<Float> diffs = new ArrayList<>();
            List<Float> featureMap = featureMaps.get(featureMapName);
            float[] dataBeforeTrain = oldFeatureMap.get(featureMapName);
            int length = dataBeforeTrain.length;
            for (int i = 0; i < length; ++i) {
                float diff = featureMap.get(i) - dataBeforeTrain[i] * (float) trainDataSize;
                diffs.add(diff);
            }
            diffFeatureMaps.put(featureMapName, diffs);
        }

        // sparse encode
        int paramNum = 0;
        for (String featureMapName : diffFeatureMaps.keySet()) {
            int weightSize = diffFeatureMaps.get(featureMapName).size();
            paramNum += weightSize;
        }
        List<Integer> maskArray = constructMaskArray(paramNum);

        Map<String, List<Float>> sparseFeatureMaps = new HashMap<>();
        int index = 0;
        for (String featureMapName : diffFeatureMaps.keySet()) {
            List<Float> sparseFeatureMap = new ArrayList<>();
            List<Float> Weight = diffFeatureMaps.get(featureMapName);
            for (Float dataValue : Weight) {
                if (maskArray.get(index) == 1) {
                    sparseFeatureMap.add(dataValue);
                }
                index += 1;
            }
            sparseFeatureMaps.put(featureMapName, sparseFeatureMap);
        }

        // quant encode
        float temp1 = (float) (1 << numBits) - 1.0f;
        float temp2 = (float) (1 << (numBits - 1));
        for (String featureMapName : sparseFeatureMaps.keySet()) {
            CompressWeight compressWeight = new CompressWeight();
            compressWeight.setWeightFullname(featureMapName);

            List<Float> sparseFeatureMap = sparseFeatureMaps.get(featureMapName);

            // get min and max value
            Float minVal = Float.MAX_VALUE;
            float maxVal = -minVal;
            for (Float value : sparseFeatureMap) {
                if (value < minVal) {
                    minVal = value;
                }
                if (value > maxVal) {
                    maxVal = value;
                }
            }
            compressWeight.setMinValue(minVal);
            compressWeight.setMaxValue(maxVal);
            float scale_value = (maxVal - minVal) / temp1 + 1e-10f;
            List<Byte> compressData = new ArrayList<>();
            for (Float aFloat : sparseFeatureMap) {
                compressData.add((byte) (Math.round((aFloat - minVal) / scale_value - temp2)));
            }
            compressWeight.setCompressData(compressData);
            compressWeights.add(compressWeight);
        }

        return compressWeights;
    }

    public List<CompressWeight> encode(Map<String, List<Float>> featureMaps, int trainDataSize) {
        byte uploadCompressType = localFLParameter.getUploadCompressType();
        if (uploadCompressType == DIFF_SPARSE_QUANT) {
            return enDiffSparseQuant(featureMaps, 8, trainDataSize);
        }
        throw new IllegalArgumentException();
    }
}