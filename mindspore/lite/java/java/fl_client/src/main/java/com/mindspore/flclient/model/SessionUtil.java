/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
import com.mindspore.lite.TrainSession;
import com.mindspore.lite.config.MSConfig;

import mindspore.schema.FeatureMap;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * session util class
 *
 * @since v1.0
 */
public class SessionUtil {
    private static final Logger logger = Logger.getLogger(SessionUtil.class.toString());

    /**
     * convert tensor to feature map
     *
     * @param tensors input tensors
     * @return feature map
     */
    public static Map<String, float[]> convertTensorToFeatures(List<MSTensor> tensors) {
        if (tensors == null) {
            logger.severe(Common.addTag("tensors cannot be null"));
            return new HashMap<>();
        }
        Map<String, float[]> features = new HashMap<>(tensors.size());
        for (MSTensor mstensor : tensors) {
            if (mstensor == null) {
                logger.severe(Common.addTag("tensors cannot be null"));
                return new HashMap<>();
            }
            features.put(mstensor.tensorName(), mstensor.getFloatData());
        }
        return features;
    }

    /**
     * get feature tensors
     *
     * @param trainSession train session
     * @return feature tensors
     */
    public static List<MSTensor> getFeatures(LiteSession trainSession) {
        if (trainSession == null) {
            return new ArrayList<MSTensor>();
        }
        return trainSession.getFeaturesMap();
    }

    /**
     * update feature tensor
     *
     * @param trainSession train session
     * @param modelName model name
     * @param featureMaps new feature map
     * @return update if success or not
     */
    public static int updateFeatures(LiteSession trainSession, String modelName, List<FeatureMap> featureMaps) {
        if (trainSession == null || featureMaps == null || modelName == null || modelName.isEmpty()) {
            logger.severe(Common.addTag("trainSession,featureMaps modelName cannot be null"));
            return -1;
        }
        List<MSTensor> tensors = new ArrayList<MSTensor>(featureMaps.size());
        for (FeatureMap newFeature : featureMaps) {
            if (newFeature == null) {
                logger.severe(Common.addTag("newFeature cannot be null"));
                return -1;
            }
            ByteBuffer by = newFeature.dataAsByteBuffer();
            ByteBuffer newData = ByteBuffer.allocateDirect(by.remaining());
            newData.order(ByteOrder.nativeOrder());
            newData.put(by);
            tensors.add(new MSTensor(newFeature.weightFullname(), newData));
        }
        boolean isSuccess = trainSession.updateFeatures(tensors);
        for (MSTensor tensor : tensors) {
            if (tensor == null) {
                logger.severe(Common.addTag("tensor cannot be null"));
                return -1;
            }
            tensor.free();
        }

        if (isSuccess) {
            trainSession.export(modelName, 0, 0);
            return 0;
        }
        return -1;
    }

    /**
     * init train session
     *
     * @param modelPath model path
     * @return train session
     */
    public static Optional<LiteSession> initSession(String modelPath) {
        if (modelPath == null) {
            logger.severe(Common.addTag("modelPath cannot be empty"));
            return Optional.empty();
        }
        MSConfig msConfig = new MSConfig();
        // arg 0: DeviceType:DT_CPU -> 0
        // arg 1: ThreadNum -> 2
        // arg 2: cpuBindMode:NO_BIND ->  0
        // arg 3: enable_fp16 -> false
        msConfig.init(0, 1, 0, false);
        LiteSession trainSession = TrainSession.createTrainSession(modelPath, msConfig, false);
        if (trainSession == null) {
            logger.severe(Common.addTag("init session failed,please check model path:" + modelPath));
            return Optional.empty();
        }
        return Optional.of(trainSession);
    }

    /**
     * free session
     *
     * @param trainSession train model owned session need to free
     */
    public static void free(LiteSession trainSession) {
        if (trainSession == null) {
            logger.severe(Common.addTag("trainSession cannot be null"));
            return;
        }
        trainSession.free();
    }
}
