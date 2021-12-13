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

package com.mindspore.flclient.example.lenet;

import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.CommonUtils;
import com.mindspore.flclient.model.Status;
import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Defining the Callback get model predict result.
 *
 * @since v1.0
 */
public class PredictCallback extends Callback {
    private final List<Integer> predictResults = new ArrayList<>();
    private final int numOfClass;
    private final int batchSize;

    /**
     * Defining a constructor of predict callback.
     */
    public PredictCallback(LiteSession session, int batchSize, int numOfClass) {
        super(session);
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
    }

    /**
     * Get predict results.
     *
     * @return predict result.
     */
    public List<Integer> getPredictResults() {
        return predictResults;
    }

    @Override
    public Status stepBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status stepEnd() {
        Optional<MSTensor> outputTensor = searchOutputsForSize(batchSize * numOfClass);
        if (!outputTensor.isPresent()) {
            return Status.FAILED;
        }
        float[] scores = outputTensor.get().getFloatData();
        for (int b = 0; b < batchSize; b++) {
            int predictIdx = CommonUtils.getMaxScoreIndex(scores, numOfClass * b, numOfClass * b + numOfClass);
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