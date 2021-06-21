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

package com.mindspore.flclient.model;

import com.mindspore.flclient.Common;
import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.config.MSConfig;
import mindspore.schema.FeatureMap;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class SessionUtil {
    private static final Logger logger = Logger.getLogger(SessionUtil.class.toString());

    public static Map<String, float[]> convertTensorToFeatures(List<MSTensor> tensors) {
        if (tensors == null) {
            logger.severe(Common.addTag("tensors cannot be null"));
            return new HashMap<>();
        }
        Map<String, float[]> features = new HashMap<>(tensors.size());
        for (MSTensor mstensor : tensors) {
            features.put(mstensor.tensorName(), mstensor.getFloatData());
        }
        return features;
    }

    public static List<MSTensor> getFeatures(LiteSession trainSession) {
        return trainSession.getFeaturesMap();
    }

    public static int updateFeatures(LiteSession trainSession, String modelName, List<FeatureMap> featureMaps) {
        if (trainSession == null || featureMaps == null || modelName.isEmpty()) {
            logger.severe(Common.addTag("trainSession,featureMaps modelName cannot be null"));
            return -1;
        }
        List<MSTensor> tensors = new ArrayList<>(featureMaps.size());
        for (FeatureMap newFeature : featureMaps) {
            ByteBuffer by = newFeature.dataAsByteBuffer();
            ByteBuffer newData = ByteBuffer.allocateDirect(by.remaining());
            newData.order(ByteOrder.nativeOrder());
            newData.put(by);
            tensors.add(new MSTensor(newFeature.weightFullname(), newData));
        }
        boolean success = trainSession.updateFeatures(tensors);
        for (MSTensor tensor : tensors) {
            tensor.free();
        }
        if (success) {
            trainSession.export(modelName,0,0);
            return 0;
        }
        return -1;
    }

    public static LiteSession initSession(String modelPath) {
        if (modelPath.isEmpty()) {
            logger.severe(Common.addTag("modelPath cannot be empty"));
            return null;
        }
        MSConfig msConfig = new MSConfig();
        // arg 0: DeviceType:DT_CPU -> 0
        // arg 1: ThreadNum -> 2
        // arg 2: cpuBindMode:NO_BIND ->  0
        // arg 3: enable_fp16 -> false
        msConfig.init(0, 1, 0, false);
        LiteSession trainSession = LiteSession.createTrainSession(modelPath, msConfig,false);
        if (trainSession == null) {
            logger.severe(Common.addTag("init session failed,please check model path:" + modelPath));
            return null;
        }
        return trainSession;
    }

    public static MSTensor searchOutputsForSize(LiteSession trainSession, int size) {
        if (trainSession == null) {
            logger.severe(Common.addTag("trainSession cannot be null"));
            return null;
        }
        Map<String, MSTensor> outputs = trainSession.getOutputMapByTensor();
        for (MSTensor tensor : outputs.values()) {
            if (tensor.elementsNum() == size) {
                return tensor;
            }
        }
        logger.severe(Common.addTag("can not find output the tensor,element num is " + size));
        return null;
    }

    public static float getLoss(LiteSession trainSession) {
        if (trainSession == null) {
            logger.severe(Common.addTag("trainSession cannot be null"));
            return Float.NaN;
        }
        MSTensor tensor = SessionUtil.searchOutputsForSize(trainSession, 1);
        if (tensor == null) {
            logger.severe(Common.addTag("cannot find loss tensor"));
            return Float.NaN;
        }
        return tensor.getFloatData()[0];
    }

    public static void free(LiteSession trainSession) {
        if (trainSession == null) {
            logger.severe(Common.addTag("trainSession cannot be null"));
            return;
        }
        trainSession.free();
    }
}
