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

import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * al infer bert for infer
 *
 * @since v1.0
 */
public class AlInferBert extends AlBert {
    private static final Logger logger = Logger.getLogger(AlInferBert.class.toString());

    private static volatile AlInferBert alInferBert;

    private AlInferBert() {
    }

    /**
     * get singleton instance
     *
     * @return singleton instance
     */
    public static AlInferBert getInstance() {
        AlInferBert localRef = alInferBert;
        if (localRef == null) {
            synchronized (AlInferBert.class) {
                localRef = alInferBert;
                if (localRef == null) {
                    alInferBert = localRef = new AlInferBert();
                }
            }
        }
        return localRef;
    }

    /**
     * init data set
     *
     * @param exampleFile example file
     * @param vocabFile vocab file
     * @param idsFile ids file
     * @param isEvalMode if in eval mod
     * @return data set size
     */
    public int initDataSet(String exampleFile, String vocabFile, String idsFile, boolean isEvalMode) {
        if (exampleFile == null || vocabFile == null || idsFile == null) {
            logger.severe(Common.addTag("dataset init failed,trainFile,idsFile,vocabFile cannot be empty"));
            return -1;
        }
        if (isEvalMode) {
            features = DataSet.init(exampleFile, vocabFile, idsFile, false, maxSeqLen);
        } else {
            features = DataSet.readInferData(exampleFile, vocabFile, idsFile, false, maxSeqLen);
        }
        return features.size();
    }

    private int[] infer() {
        boolean isSuccess = trainSession.eval();
        if (!isSuccess) {
            logger.severe(Common.addTag("trainSession switch eval mode failed"));
            return new int[0];
        }
        int[] predictLabels = new int[features.size()];
        for (int j = 0; j < batchNum; j++) {
            List<Integer> labels = fillModelInput(j, false);
            if (labels == null) {
                logger.severe(Common.addTag("fill model input failed"));
                return new int[0];
            }
            isSuccess = trainSession.runGraph();
            if (!isSuccess) {
                logger.severe(Common.addTag("run graph failed"));
                return new int[0];
            }
            int[] batchLabels = getBatchLabel();
            System.arraycopy(batchLabels, 0, predictLabels, j * batchSize, batchSize);
        }
        return predictLabels;
    }

    /**
     * infer model
     *
     * @param modelPath model file path
     * @param dataFile data file
     * @param vocabFile vocab file
     * @param idsFile ids file
     * @return infer result
     */
    public int[] inferModel(String modelPath, String dataFile, String vocabFile, String idsFile) {
        if (modelPath == null || vocabFile == null || idsFile == null || dataFile == null) {
            logger.severe(Common.addTag("dataset init failed,modelPath,idsFile,vocabFile,dataFile cannot be empty"));
            return new int[0];
        }
        logger.info(Common.addTag("Infer model," + modelPath + ",Data file," + dataFile + ",vocab file," + vocabFile +
                ",idsFile," + idsFile));
        int status = initSessionAndInputs(modelPath, false);
        if (status == -1) {
            logger.severe(Common.addTag("init session and inputs failed"));
            return new int[0];
        }
        int inferSize = initDataSet(dataFile, vocabFile, idsFile, false);
        if (inferSize == 0) {
            logger.severe(Common.addTag("infer size should more than 0"));
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
