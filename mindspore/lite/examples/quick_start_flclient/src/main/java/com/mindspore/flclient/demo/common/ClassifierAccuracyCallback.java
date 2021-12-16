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

package com.mindspore.flclient.demo.common;

import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.CommonUtils;
import com.mindspore.flclient.model.Status;
import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;

import java.util.List;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * Defining the Callback calculate classifier model.
 *
 * @since v1.0
 */
public class ClassifierAccuracyCallback extends Callback {
    private static final Logger LOGGER = Logger.getLogger(ClassifierAccuracyCallback.class.toString());
    private final int numOfClass;
    private final int batchSize;
    private final List<Integer> targetLabels;
    private float accuracy;

    /**
     * Defining a constructor of  ClassifierAccuracyCallback.
     */
    public ClassifierAccuracyCallback(LiteSession session, int batchSize, int numOfClass, List<Integer> targetLabels) {
        super(session);
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
        this.targetLabels = targetLabels;
    }

    /**
     * Get eval accuracy.
     *
     * @return accuracy.
     */
    public float getAccuracy() {
        return accuracy;
    }

    @Override
    public Status stepBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status stepEnd() {
        Status status = calAccuracy();
        if (status != Status.SUCCESS) {
            return status;
        }
        steps++;
        return Status.SUCCESS;
    }

    @Override
    public Status epochBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status epochEnd() {
        LOGGER.info("average accuracy:" + steps + ",acc is:" + accuracy / steps);
        accuracy = accuracy / steps;
        steps = 0;
        return Status.SUCCESS;
    }

    private Status calAccuracy() {
        if (targetLabels == null || targetLabels.isEmpty()) {
            LOGGER.severe("labels cannot be null");
            return Status.NULLPTR;
        }
        Optional<MSTensor> outputTensor = searchOutputsForSize(batchSize * numOfClass);
        if (!outputTensor.isPresent()) {
            return Status.NULLPTR;
        }
        float[] scores = outputTensor.get().getFloatData();
        int hitCounts = 0;
        for (int b = 0; b < batchSize; b++) {
            int predictIdx = CommonUtils.getMaxScoreIndex(scores, numOfClass * b, numOfClass * b + numOfClass);
            if (targetLabels.get(b + steps * batchSize) == predictIdx) {
                hitCounts += 1;
            }
        }
        accuracy += ((float) (hitCounts) / batchSize);
        LOGGER.info("steps:" + steps + ",acc is:" + (float) (hitCounts) / batchSize);
        return Status.SUCCESS;
    }

}
