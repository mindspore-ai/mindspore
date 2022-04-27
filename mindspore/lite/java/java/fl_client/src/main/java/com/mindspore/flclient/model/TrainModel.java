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

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * ad train bert for train
 *
 * @since v1.0
 */
public abstract class TrainModel {
    private static final Logger logger = Logger.getLogger(TrainModel.class.toString());

    LiteSession trainSession;

    int batchSize = 0;

    int trainSampleSize = 0;

    int batchNum = 0;

    int numOfClass = 0;

    int padSize = 0;

    /**
     * init session and model inputs
     *
     * @param modelPath   model file path
     * @param isTrainMode if train mod
     * @return status if success return 0 else -1
     */
    public abstract int initSessionAndInputs(String modelPath, boolean isTrainMode);

    /**
     * fill model input data
     *
     * @param batchIdx    batch index
     * @param isTrainMode if train mod
     * @return model input labels
     */
    public abstract List<Integer> fillModelInput(int batchIdx, boolean isTrainMode);

    /**
     * pad samples
     *
     * @return status if success return 0 else -1
     */
    public abstract int padSamples();

    /**
     * train model
     *
     * @param modelPath model file path
     * @param epochs    train epoch number
     * @return status if success return 0 else -1
     */
    public int trainModel(String modelPath, int epochs) {
        if (modelPath == null) {
            logger.severe(Common.addTag("model path cannot be empty"));
            return -1;
        }
        if (epochs <= 0) {
            logger.severe(Common.addTag("epochs cannot smaller than 0"));
            return -1;
        }
        int status = padSamples();
        if (status != 0) {
            logger.severe(Common.addTag("train model failed"));
            return -1;
        }
        status = trainLoop(epochs);
        if (status == -1) {
            logger.severe(Common.addTag("train loop failed"));
            return -1;
        }
        boolean isSuccess = trainSession.export(modelPath, 0, 1);
        if (!isSuccess) {
            logger.severe(Common.addTag("save model failed"));
            return -1;
        }
        return 0;
    }

    private int trainLoop(int epochs) {
        boolean isTrain = trainSession.train();
        if (!isTrain) {
            logger.severe(Common.addTag("trainsession set train failed"));
            return -1;
        }
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++) {
            float sumLossPerEpoch = 0.0f;
            for (int j = 0; j < batchNum; j++) {
                List<Integer> labels = fillModelInput(j, true);
                if (labels == null) {
                    logger.severe(Common.addTag("train model fill model input failed"));
                    return -1;
                }
                boolean isSuccess = trainSession.runGraph();
                if (!isSuccess) {
                    logger.severe(Common.addTag("run graph failed"));
                    return -1;
                }
                float loss = getLoss(trainSession);
                if (Float.isNaN(loss)) {
                    logger.severe(Common.addTag("loss is nan"));
                    return -1;
                }
                sumLossPerEpoch += loss;
                logger.info(Common.addTag("batch:" + j + ",loss:" + loss));
            }
            logger.info(Common.addTag("----------epoch:" + i + ",mean loss:" + sumLossPerEpoch / batchNum +
                    "----------"));
            long endTime = System.currentTimeMillis();
            logger.info(Common.addTag("total train time:" + (endTime - startTime) + "ms"));
        }
        return 0;
    }

    private float getLoss(LiteSession trainSession) {
        Optional<MSTensor> tensor = searchOutputsForSize(trainSession, 1);
        if (!tensor.isPresent()) {
            logger.severe(Common.addTag("cannot find loss tensor"));
            return Float.NaN;
        }
        return tensor.get().getFloatData()[0];
    }

    private Optional<MSTensor> searchOutputsForSize(LiteSession trainSession, int size) {
        if (trainSession == null) {
            logger.severe(Common.addTag("trainSession cannot be null"));
            return Optional.empty();
        }
        Map<String, MSTensor> outputs = trainSession.getOutputMapByTensor();
        for (MSTensor tensor : outputs.values()) {
            if (tensor == null) {
                logger.severe(Common.addTag("tensor cannot be null"));
                return Optional.empty();
            }
            if (tensor.elementsNum() == size) {
                return Optional.of(tensor);
            }
        }
        logger.severe(Common.addTag("can not find output the tensor,element num is " + size));
        return Optional.empty();
    }

    private int calAccuracy(List<Integer> labels, int numOfClass, int padSize) {
        if (labels == null || labels.isEmpty()) {
            logger.severe(Common.addTag("labels cannot be null"));
            return -1;
        }
        Optional<MSTensor> outputTensor = searchOutputsForSize(trainSession, batchSize * numOfClass);
        if (!outputTensor.isPresent()) {
            return Integer.MAX_VALUE;
        }
        float[] scores = outputTensor.get().getFloatData();
        int accuracy = 0;
        boolean isPad = padSize != 0;
        for (int b = 0; b < batchSize; b++) {
            if (isPad && b == batchSize - padSize) {
                return accuracy;
            }
            int predictIdx = getPredictLabel(scores, numOfClass * b, numOfClass * b + numOfClass);
            if (labels.get(b) == predictIdx) {
                accuracy += 1;
            }
        }
        return accuracy;
    }

    /**
     * get batch labels
     *
     * @return batch label result
     */
    public int[] getBatchLabel() {
        Optional<MSTensor> outputTensor = searchOutputsForSize(trainSession, batchSize * numOfClass);
        if (!outputTensor.isPresent()) {
            return new int[0];
        }
        int[] inferLabels = new int[batchSize];
        float[] scores = outputTensor.get().getFloatData();
        for (int b = 0; b < batchSize; b++) {
            inferLabels[b] = getPredictLabel(scores, numOfClass * b, numOfClass * b + numOfClass);
        }
        return inferLabels;
    }

    /**
     * eval model
     *
     * @return eval result
     */
    public float evalModel() {
        int ret = padSamples();
        if (ret != 0) {
            logger.severe(Common.addTag("eval model failed"));
            return Float.NaN;
        }
        boolean isSuccess = trainSession.eval();
        if (!isSuccess) {
            logger.severe(Common.addTag("train session switch eval mode failed"));
            return Float.NaN;
        }

        float totalRightPredicts = 0.0f;
        for (int j = 0; j < batchNum; j++) {
            List<Integer> labels = fillModelInput(j, false);
            if (labels == null) {
                logger.severe(Common.addTag("train model fill model input failed"));
                return Float.NaN;
            }
            long startTime = System.currentTimeMillis();
            isSuccess = trainSession.runGraph();
            if (!isSuccess) {
                logger.severe(Common.addTag("run graph failed"));
                return Float.NaN;
            }
            long endTime = System.currentTimeMillis();
            logger.info(Common.addTag("run graph time cost:" + (endTime - startTime)));
            int batchPadSize = j == batchNum - 1 ? padSize : 0;
            if (batchSize <= batchPadSize) {
                logger.severe(Common.addTag("pad size error"));
                return Float.NaN;
            }
            float curAcc = calAccuracy(labels, numOfClass, batchPadSize);
            if (curAcc == Integer.MAX_VALUE) {
                logger.severe(Common.addTag("cur acc is too big"));
                return Float.NaN;
            }
            totalRightPredicts += curAcc;
            logger.info(Common.addTag("batch num:" + j + ",acc is:" + curAcc / (batchSize - batchPadSize)));
        }
        if (trainSampleSize - padSize <= 0) {
            logger.severe(Common.addTag("train sample size cannot less than pad size"));
            return Float.NaN;
        }
        float totalAccuracy = totalRightPredicts / (trainSampleSize - padSize);
        logger.info(Common.addTag("total acc:" + totalAccuracy));
        return totalAccuracy;
    }

    private int getPredictLabel(float[] scores, int start, int end) {
        if (scores == null || scores.length == 0) {
            logger.severe(Common.addTag("scores cannot be empty"));
            return -1;
        }
        if (start >= scores.length || start < 0 || end > scores.length || end < 0) {
            logger.severe(Common.addTag("start,end cannot out of scores length"));
            return -1;
        }
        float maxScore = scores[start];
        int maxIdx = start;
        for (int i = start; i < end; i++) {
            if (scores[i] > maxScore) {
                maxIdx = i;
                maxScore = scores[i];
            }
        }
        return maxIdx - start;
    }

    /**
     * set batch size
     *
     * @param batchSize batch size
     * @return status if success return 0 else -1
     */
    public int setBatchSize(int batchSize) {
        if (batchSize <= 0) {
            logger.severe(Common.addTag("batch size should more than 0"));
            return -1;
        }
        this.batchSize = batchSize;
        return 0;
    }

    /**
     * get train session
     *
     * @return train session
     */
    public LiteSession getTrainSession() {
        return trainSession;
    }
}
