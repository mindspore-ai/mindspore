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

import com.mindspore.Model;
import com.mindspore.flclient.Common;
import com.mindspore.flclient.common.FLLoggerGenerater;
import com.mindspore.MSTensor;

import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * Defining the Callback get model loss.
 *
 * @since v1.0
 */
public class LossCallback extends Callback {
    private static final Logger logger = FLLoggerGenerater.getModelLogger(LossCallback.class.toString());

    private float lossSum = 0.0f;

    private float uploadLoss = 0.0f;

    /**
     * Defining a constructor of loss callback.
     */
    public LossCallback(com.mindspore.Model model) {
        super(model);
    }

    @Override
    public Status stepBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status stepEnd() {
        Map<String, float[]> outputs = getOutputsBySize(1);
        if (outputs.isEmpty()) {
            logger.severe("cannot find loss tensor");
            return Status.NULLPTR;
        }
        Map.Entry<String, float[]> first = outputs.entrySet().iterator().next();
        if (first.getValue().length < 1 || Float.isNaN(first.getValue()[0])) {
            logger.severe("loss is nan");
            return Status.FAILED;
        }
        float loss = first.getValue()[0];
        logger.info("batch:" + steps + ",loss:" + loss);
        lossSum += loss;
        steps++;
        return Status.SUCCESS;
    }

    @Override
    public Status epochBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status epochEnd() {
        logger.info("----------epoch:" + epochs + ",average loss:" + lossSum / steps + "----------");
        setUploadLoss(lossSum / steps);
        steps = 0;
        epochs++;
        lossSum = 0.0f;
        return Status.SUCCESS;
    }
    public float getUploadLoss() {
        return uploadLoss;
    }

    public void setUploadLoss(float uploadLoss) {
        this.uploadLoss = uploadLoss;
    }
}