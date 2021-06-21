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

import java.util.Arrays;
import java.util.logging.Logger;

public class AdInferBert extends AdBert {
    private static final Logger logger = Logger.getLogger(AdInferBert.class.toString());

    private static volatile AdInferBert adInferBert;

    public static AdInferBert getInstance() {
        AdInferBert localRef = adInferBert;
        if (localRef == null) {
            synchronized (AdInferBert.class) {
                localRef = adInferBert;
                if (localRef == null) {
                    adInferBert = localRef = new AdInferBert();
                }
            }
        }
        return localRef;
    }

    public int initDataSet(String exampleFile, String vocabFile, String idsFile, boolean evalMod) {
        if (evalMod) {
            features = DataSet.init(exampleFile, vocabFile, idsFile, false);
        } else {
            features = DataSet.readInferData(exampleFile, vocabFile, idsFile, false);
        }
        if (features == null) {
            logger.severe(Common.addTag("features cannot be null"));
            return -1;
        }
        return features.size();
    }

    private int[] infer() {
        boolean success = trainSession.eval();
        if (!success) {
            logger.severe(Common.addTag("trainSession switch eval mode failed"));
            return new int[0];
        }
        int[] predictLabels = new int[features.size()];
        for (int j = 0; j < batchNum; j++) {
            fillModelInput(j, false);
            success = trainSession.runGraph();
            if (!success) {
                logger.severe(Common.addTag("run graph failed"));
                return new int[0];
            }
            int[] batchLabels = getBatchLabel();
            System.arraycopy(batchLabels, 0, predictLabels, j * batchSize, batchSize);
        }
        return predictLabels;
    }

    public int[] inferModel(String modelPath, String dataFile, String vocabFile, String idsFile) {
        logger.info(Common.addTag("Infer model," + modelPath + ",Data file," + dataFile + ",vocab file," + vocabFile + ",idsFile," + idsFile));
        int inferSize = initDataSet(dataFile, vocabFile, idsFile, false);
        if (inferSize == 0) {
            logger.severe(Common.addTag("infer size should more than 0"));
            return new int[0];
        }
        int status = initSessionAndInputs(modelPath, false);
        if (status == -1) {
            logger.severe(Common.addTag("init session and inputs failed"));
            return new int[0];
        }
        status = padSamples();
        if (status == -1) {
            logger.severe(Common.addTag("infer model failed"));
            return new int[0];
        }
        if (batchSize <= 0) {
            logger.severe(Common.addTag("batch size must bigger than 0"));
            return new int[0];
        }
        batchNum = features.size() / batchSize;
        int[] predictLabels = infer();
        if (predictLabels.length == 0) {
            return new int[0];
        }
        return Arrays.copyOfRange(predictLabels, 0, inferSize);
    }
}
