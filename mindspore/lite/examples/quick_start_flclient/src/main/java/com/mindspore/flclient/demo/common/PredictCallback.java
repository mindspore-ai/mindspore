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

import com.mindspore.Model;
import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.Status;
import com.mindspore.MSTensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * Defining the Callback get model predict result.
 *
 * @since v1.0
 */
public class PredictCallback extends Callback {
    private static final Logger LOGGER = Logger.getLogger(PredictCallback.class.toString());
    private final List<Object> predictResults = new ArrayList<>();
    private final int numOfClass;
    private final int batchSize;

    /**
     * Defining a constructor of predict callback.
     */
    public PredictCallback(Model model, int batchSize, int numOfClass) {
        super(model);
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
    }

    public static int getMaxScoreIndex(float[] scores, int start, int end) {
        if (scores != null && scores.length != 0) {
            if (start < scores.length && start >= 0 && end <= scores.length && end >= 0) {
                float maxScore = scores[start];
                int maxIdx = start;

                for (int i = start; i < end; ++i) {
                    if (scores[i] > maxScore) {
                        maxIdx = i;
                        maxScore = scores[i];
                    }
                }

                return maxIdx - start;
            } else {
                LOGGER.severe("start,end cannot out of scores length");
                return -1;
            }
        } else {
            LOGGER.severe("scores cannot be empty");
            return -1;
        }
    }

    /**
     * Get predict results.
     *
     * @return predict result.
     */
    public List<Object> getPredictResults() {
        return predictResults;
    }

    @Override
    public Status stepBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status stepEnd() {
        Map<String, float[]> outputs = getOutputsBySize(batchSize * numOfClass);
        if (outputs.isEmpty()) {
            LOGGER.severe("cannot find loss tensor");
            return Status.FAILED;
        }
        Map.Entry<String, float[]> first = outputs.entrySet().iterator().next();
        float[] scores = first.getValue();
        for (int b = 0; b < batchSize; b++) {
            int predictIdx = getMaxScoreIndex(scores, numOfClass * b, numOfClass * b + numOfClass);
            predictResults.add(predictIdx);
        }
        return Status.SUCCESS;
    }

    @Override
    public Status epochBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status epochEnd() {
        return Status.SUCCESS;
    }
}
