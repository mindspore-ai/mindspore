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

import java.util.logging.Logger;

/**
 * al train bert for train
 *
 * @since v1.0
 */
public class AlTrainBert extends AlBert {
    private static final Logger logger = Logger.getLogger(AlTrainBert.class.toString());

    private static volatile AlTrainBert alTrainBert;

    private AlTrainBert() {
    }

    /**
     * get singleton instance
     *
     * @return singleton instance
     */
    public static AlTrainBert getInstance() {
        AlTrainBert localRef = alTrainBert;
        if (localRef == null) {
            synchronized (AlTrainBert.class) {
                localRef = alTrainBert;
                if (localRef == null) {
                    alTrainBert = localRef = new AlTrainBert();
                }
            }
        }
        return localRef;
    }

    /**
     * init data set
     *
     * @param dataFile data file
     * @param vocabFile vocab file
     * @param idsFile ids file
     * @return data set size
     */
    public int initDataSet(String dataFile, String vocabFile, String idsFile) {
        if (dataFile == null || vocabFile == null || idsFile == null) {
            logger.severe(Common.addTag("dataFile,idsFile,vocabFile cannot be empty"));
            return -1;
        }
        features = DataSet.init(dataFile, vocabFile, idsFile, true, maxSeqLen);
        return features.size();
    }
}
