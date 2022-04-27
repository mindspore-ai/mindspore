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

import java.nio.ByteBuffer;
import java.util.List;
import java.util.logging.Logger;

/**
 * Defining the dataset base loss.
 *
 * @since v1.0
 */
public abstract class DataSet {
    private static final Logger logger = Logger.getLogger(DataSet.class.toString());

    /**
     * dataset sample size.
     */
    public int sampleSize;

    /**
     * batch nums each epoch.
     */
    public int batchNum;

    /**
     * batch size.
     */
    public int batchSize;

    /**
     * Fill inputs buffer.
     *
     * @param inputsBuffer to be filled buffer.
     * @param batchIdx     batch index.
     */
    public abstract void fillInputBuffer(List<ByteBuffer> inputsBuffer, int batchIdx);

    /**
     * Shuffle dataset.
     */
    public abstract void shuffle();

    /**
     * Padding dataset.
     */
    public abstract void padding();

    /**
     * Dataset preprocess.
     *
     * @param files data files.
     * @return preprocess status.
     */
    public abstract Status dataPreprocess(List<String> files);

    /**
     * Init dataset.
     *
     * @param files data files.
     * @return init status.
     */
    public Status init(List<String> files) {
        Status status = dataPreprocess(files);
        if (status != Status.SUCCESS) {
            logger.severe(Common.addTag("data preprocess failed"));
            return status;
        }
        shuffle();
        return status;
    }
}