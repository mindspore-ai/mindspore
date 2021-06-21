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

import java.util.logging.Logger;

public class AdTrainBert extends AdBert {
    private static final Logger logger = Logger.getLogger(AdTrainBert.class.toString());

    private static volatile AdTrainBert adTrainBert;

    public static AdTrainBert getInstance() {
        AdTrainBert localRef = adInferBert;
        if (localRef == null) {
            synchronized (AdTrainBert.class) {
                localRef = adTrainBert;
                if (localRef == null) {
                    adTrainBert = localRef = new AdTrainBert();
                }
            }
        }
        return localRef;
    }

    public int initDataSet(String dataFile, String vocabFile, String idsFile) {
        features = DataSet.init(dataFile, vocabFile, idsFile, true);
        if (features == null) {
            logger.severe(Common.addTag("features cannot be null"));
            return -1;
        }
        return features.size();
    }
}


