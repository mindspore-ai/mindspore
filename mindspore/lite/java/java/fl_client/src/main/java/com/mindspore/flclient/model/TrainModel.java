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
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.LiteSession;

import java.util.List;
import java.util.logging.Logger;

public abstract class TrainModel {

    private static final Logger logger = Logger.getLogger(TrainModel.class.toString());

    LiteSession trainSession;

    int batchSize = 0;

    int trainSampleSize = 0;

    int batchNum = 0;

    int numOfClass = 0;

    int padSize = 0;

    public abstract int initSessionAndInputs(String modelPath, boolean trainMod);

    public abstract List<Integer> fillModelInput(int batchIdx, boolean trainMod);

    public abstract int padSamples();

    public int trainModel(String modelPath, int epochs) {
        if (modelPath.isEmpty()) {
            logger.severe(Common.addTag("model path cannot be empty"));
            return -1;
        }
        if (epochs <= 0) {
            logger.severe(Common.addTag("epochs cannot smaller than 0"));
            return -1;
        }
        if (batchSize <= 0) {
            logger.severe(Common.addTag("batch size must bigger than 0"));
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
        boolean success = trainSession.export(modelPath,0,0);
        if (!success) {
            logger.severe(Common.addTag("save model failed"));
            return -1;
        }
        return 0;
    }

    private int trainLoop(int epochs) {
        if (batchNum <= 0) {
            logger.severe(Common.addTag("batch num must bigger than 0"));
            return -1;
        }
        trainSession.train();
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++) {
            float sumLossPerEpoch = 0.0f;
            for (int j = 0; j < batchNum; j++) {
                fillModelInput(j, true);
                boolean success = trainSession.runGraph();
                if (!success) {
                    logger.severe(Common.addTag("run graph failed"));
                    return -1;
                }
                float loss = SessionUtil.getLoss(trainSession);
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

    public float evalModel() {
        int ret = padSamples();
        if (ret != 0) {
            logger.severe(Common.addTag("eval model failed"));
            return Float.NaN;
        }
        boolean success = trainSession.eval();
        if (!success) {
            logger.severe(Common.addTag("train session switch eval mode failed"));
            return Float.NaN;
        }
        float totalRightPredicts = 0.0f;
        for (int j = 0; j < batchNum; j++) {
            List<Integer> labels = fillModelInput(j, false);
            long startTime = System.currentTimeMillis();
            success = trainSession.runGraph();
            if (!success) {
                logger.severe(Common.addTag("run graph failed"));
                return Float.NaN;
            }
            long endTime = System.currentTimeMillis();
            logger.info(Common.addTag("run graph time cost:" + (endTime - startTime)));
            int batchPadSize = j == batchNum - 1 ? padSize : 0;
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


    // if sampale is padded.need rm pad samples
    private int calAccuracy(List<Integer> labels, int numOfClass, int padSize) {
        if (labels == null) {
            logger.severe(Common.addTag("labels cannot be null"));
            return -1;
        }
        MSTensor outputTensor = SessionUtil.searchOutputsForSize(trainSession, batchSize * numOfClass);
        if (outputTensor == null) {
            return Integer.MAX_VALUE;
        }
        float[] scores = outputTensor.getFloatData();
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

    public int[] getBatchLabel() {
        MSTensor outputTensor = SessionUtil.searchOutputsForSize(trainSession, batchSize * numOfClass);
        if (outputTensor == null) {
            return new int[0];
        }
        int[] inferLabels = new int[batchSize];
        float[] scores = outputTensor.getFloatData();
        for (int b = 0; b < batchSize; b++) {
            inferLabels[b] = getPredictLabel(scores, numOfClass * b, numOfClass * b + numOfClass);
        }
        return inferLabels;
    }

    private int getPredictLabel(float[] scores, int start, int end) {
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

    public int setBatchSize(int batchSize) {
        if (batchSize <= 0) {
            logger.severe(Common.addTag("batch size should more than 0"));
            return -1;
        }
        this.batchSize = batchSize;
        return 0;
    }

    public LiteSession getTrainSession() {
        return trainSession;
    }
}
