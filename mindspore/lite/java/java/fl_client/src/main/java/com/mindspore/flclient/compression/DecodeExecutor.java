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

import com.google.flatbuffers.FlatBufferBuilder;

import com.mindspore.flclient.Common;
import com.mindspore.flclient.StartFLJob;
import com.mindspore.flclient.common.FLLoggerGenerater;
import mindspore.schema.CompressFeatureMap;
import mindspore.schema.FeatureMap;
import mindspore.schema.CompressType;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import static mindspore.schema.CompressType.QUANT;
/**
 * Compress Executor
 *
 * @since 2021-12-21
 */
public class DecodeExecutor {
    private static final Logger LOGGER = FLLoggerGenerater.getModelLogger(DecodeExecutor.class.toString());

    static public List<FeatureMap> deCompressWeight(byte compressType, List<CompressFeatureMap> compressFeatureMapList) {
        if (!CompressMode.COMPRESS_TYPE_MAP.containsKey(compressType)) {
            return new ArrayList<>();
        }
        LOGGER.info("[deCompressWeight] create " + CompressType.name(compressType) + " feature map.");
        int num_bits = CompressMode.COMPRESS_TYPE_MAP.get(compressType);
        if (compressType == QUANT) {
            return deCompressQuantMinMax(compressFeatureMapList, num_bits);
        }
        return new ArrayList<>();
    }

    static private List<FeatureMap> deCompressQuantMinMax(List<CompressFeatureMap> compressFeatureMapList, int num_bits) {
        float temp1 = (float) (Math.pow(2, num_bits) - 1);
        float temp2 = (float) Math.pow(2, num_bits - 1);

        Map<String, float[]> deCompressFeatureMaps = new HashMap<>();
        int compressFeatureMapLength = compressFeatureMapList.size();
        for (int i = 0; i < compressFeatureMapLength; i++) {
            CompressFeatureMap compressFeatureMap = compressFeatureMapList.get(i);
            String weightName = compressFeatureMap.weightFullname();
            int compressDataLength = compressFeatureMap.compressDataLength();
            List<Byte> compressWeightList = new ArrayList<>();
            for (int j = 0; j < compressDataLength; j++) {
                compressWeightList.add(compressFeatureMap.compressData(j));
            }
            float minVal = compressFeatureMap.minVal();
            float maxVal = compressFeatureMap.maxVal();
            float scale_value = (float) ((maxVal - minVal) / temp1 + 1e-10);
            float[] params = new float[compressWeightList.size()];
            for (int j = 0; j < params.length; j++) {
                float val = (compressWeightList.get(j).intValue() + temp2) * scale_value + minVal;
                params[j] = val;
            }
            deCompressFeatureMaps.put(weightName, params);
        }

        List<FeatureMap> featureMaps = new ArrayList<>();
        for (String weightName : deCompressFeatureMaps.keySet()) {
            FlatBufferBuilder builder = new FlatBufferBuilder(0);
            int weightFullnameOffset = builder.createString(weightName);
            float[] data = deCompressFeatureMaps.get(weightName);
            int dataOffset = FeatureMap.createDataVector(builder, data);

            FeatureMap.startFeatureMap(builder);
            FeatureMap.addWeightFullname(builder, weightFullnameOffset);
            FeatureMap.addData(builder, dataOffset);

            int orc = FeatureMap.endFeatureMap(builder);
            builder.finish(orc);
            ByteBuffer buf = builder.dataBuffer();
            FeatureMap featureMap = FeatureMap.getRootAsFeatureMap(buf);

            featureMaps.add(featureMap);
        }
        return featureMaps;
    }

    static public FeatureMap quantDeCompress(CompressFeatureMap compressFeature) {
        int num_bits = CompressMode.COMPRESS_TYPE_MAP.get(QUANT);
        float temp1 = (float) (Math.pow(2, num_bits) - 1);
        float temp2 = (float) Math.pow(2, num_bits - 1);

        String weightName = compressFeature.weightFullname();
        int compressDataLength = compressFeature.compressDataLength();
        List<Byte> compressWeightList = new ArrayList<>();
        for (int j = 0; j < compressDataLength; j++) {
            compressWeightList.add(compressFeature.compressData(j));
        }
        float minVal = compressFeature.minVal();
        float maxVal = compressFeature.maxVal();
        float scale_value = (float) ((maxVal - minVal) / temp1 + 1e-10);
        float[] params = new float[compressWeightList.size()];
        for (int j = 0; j < params.length; j++) {
            float val = (compressWeightList.get(j).intValue() + temp2) * scale_value + minVal;
            params[j] = val;
        }
        FlatBufferBuilder builder = new FlatBufferBuilder(0);
        int weightFullnameOffset = builder.createString(weightName);
        int dataOffset = FeatureMap.createDataVector(builder, params);
        FeatureMap.startFeatureMap(builder);
        FeatureMap.addWeightFullname(builder, weightFullnameOffset);
        FeatureMap.addData(builder, dataOffset);

        int orc = FeatureMap.endFeatureMap(builder);
        builder.finish(orc);
        ByteBuffer buf = builder.dataBuffer();
        FeatureMap featureMap = FeatureMap.getRootAsFeatureMap(buf);
        return featureMap;
    }
}