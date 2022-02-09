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

import java.util.Optional;
import java.util.logging.Logger;

/**
 * Defining the Callback get model loss.
 *
 * @since v1.0
 */
public class LossCallback extends Callback {
    private static final Logger logger = Logger.getLogger(LossCallback.class.toString());

    private float lossSum = 0.0f;

    private float uploadLoss = 0.0f;

    /**
     * Defining a constructor of loss callback.
     */
    public LossCallback(LiteSession session) {
        super(session);
    }

    @Override
    public Status stepBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status stepEnd() {
        Optional<MSTensor> tensor = searchOutputsForSize(1);
        if (!tensor.isPresent()) {
            logger.severe(Common.addTag("cannot find loss tensor"));
            return Status.NULLPTR;
        }
        float loss = tensor.get().getFloatData()[0];
        if (Float.isNaN(loss)) {
            logger.severe(Common.addTag("loss is nan"));
            return Status.FAILED;
        }
        logger.info(Common.addTag("batch:" + steps + ",loss:" + loss));
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
        logger.info(Common.addTag("----------epoch:" + epochs + ",average loss:" + lossSum / steps + "----------"));
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