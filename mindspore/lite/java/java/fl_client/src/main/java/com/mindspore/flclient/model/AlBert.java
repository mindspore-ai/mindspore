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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * albert
 *
 * @since v1.0
 */
public class AlBert extends TrainModel {
    private static final Logger logger = Logger.getLogger(AlBert.class.toString());

    private static final int NUM_OF_CLASS = 4;

    private static final int TRAIN_BERT_INPUTS = 4;

    private static final int EVAL_BERT_INPUTS = 3;

    List<Feature> features;

    int maxSeqLen = 16;

    private int dataSize;

    private ByteBuffer inputIdBufffer;

    private ByteBuffer tokenIdBufffer;

    private ByteBuffer maskIdBufffer;

    private ByteBuffer labelIdBufffer;

    private void fillInputBuffer(int batchIdx, boolean isTrainMode, List<Integer> labels) {
        for (int i = 0; i < batchSize; i++) {
            Feature feature = features.get(batchIdx * batchSize + i);
            for (int j = 0; j < dataSize; j++) {
                inputIdBufffer.putInt(feature.inputIds[j]);
                tokenIdBufffer.putInt(feature.tokenIds[j]);
                maskIdBufffer.putInt(feature.inputMasks[j]);
            }
            if (!isTrainMode) {
                labels.add(feature.labelIds);
            } else {
                labelIdBufffer.putInt(feature.labelIds);
            }
        }
    }

    @Override
    public int initSessionAndInputs(String modelPath, boolean isTrainMode) {
        if (modelPath == null) {
            logger.severe(Common.addTag("session init failed"));
            return -1;
        }
        int ret = -1;
        Optional<LiteSession> optTrainSession = SessionUtil.initSession(modelPath);
        if (!optTrainSession.isPresent()) {
            logger.severe(Common.addTag("session init failed"));
            return -1;
        }
        trainSession = optTrainSession.get();
        List<MSTensor> inputs = trainSession.getInputs();
        if (inputs.size() < 1) {
            logger.severe(Common.addTag("inputs size error"));
            return ret;
        }
       MSTensor inputIdTensor = trainSession.getInputsByTensorName("input_ids");
        if (inputIdTensor == null) {
            logger.severe(Common.addTag("labelId tensor is null"));
            return ret;
        }
        int inputSize = inputIdTensor.elementsNum();
        batchSize = inputIdTensor.getShape()[0];
        if (batchSize <= 0) {
            logger.severe(Common.addTag("batch size should bigger than 0"));
            return ret;
        }
        dataSize = inputSize / batchSize;
        maxSeqLen = dataSize;
        // tokenId,inputId,maskId has same size
        inputIdBufffer = ByteBuffer.allocateDirect(inputSize * Integer.BYTES);
        tokenIdBufffer = ByteBuffer.allocateDirect(inputSize * Integer.BYTES);
        maskIdBufffer = ByteBuffer.allocateDirect(inputSize * Integer.BYTES);
        inputIdBufffer.order(ByteOrder.nativeOrder());
        tokenIdBufffer.order(ByteOrder.nativeOrder());
        maskIdBufffer.order(ByteOrder.nativeOrder());
        if (isTrainMode) {
            labelIdBufffer = ByteBuffer.allocateDirect(batchSize * Integer.BYTES);
            labelIdBufffer.order(ByteOrder.nativeOrder());
        }
        numOfClass = NUM_OF_CLASS;
        logger.info(Common.addTag("init session and input success"));
        return 0;
    }

    @Override
    public List<Integer> fillModelInput(int batchIdx, boolean isTrainMode) {
        inputIdBufffer.clear();
        tokenIdBufffer.clear();
        maskIdBufffer.clear();
        if (isTrainMode) {
            labelIdBufffer.clear();
        }
        List<Integer> labels = new ArrayList<>();
        if ((batchIdx + 1) * batchSize - 1 >= features.size()) {
            logger.severe(Common.addTag("fill model input failed"));
            return new ArrayList<>();
        }
        fillInputBuffer(batchIdx, isTrainMode, labels);
        List<MSTensor> inputs = trainSession.getInputs();
        if (inputs.size() != TRAIN_BERT_INPUTS && inputs.size() != EVAL_BERT_INPUTS) {
            logger.severe(Common.addTag("bert input size error"));
            return new ArrayList<>();
        }
        MSTensor labelIdTensor;
        MSTensor tokenIdTensor;
        MSTensor inputIdTensor;
        MSTensor maskIdTensor;
        if (isTrainMode) {
            if (inputs.size() != TRAIN_BERT_INPUTS) {
                logger.severe(Common.addTag("train bert input size error"));
                return new ArrayList<>();
            }
            labelIdTensor = trainSession.getInputsByTensorName("label_ids");
            tokenIdTensor = trainSession.getInputsByTensorName("token_type_id");
            inputIdTensor = trainSession.getInputsByTensorName("input_ids");
            maskIdTensor = trainSession.getInputsByTensorName("input_mask");
            labelIdTensor.setData(labelIdBufffer);
        } else {
            if (inputs.size() != EVAL_BERT_INPUTS) {
                logger.severe(Common.addTag("eval bert input size error"));
                return new ArrayList<>();
            }
            tokenIdTensor = trainSession.getInputsByTensorName("token_type_id");
            inputIdTensor = trainSession.getInputsByTensorName("input_ids");
            maskIdTensor = trainSession.getInputsByTensorName("input_mask");
        }
        tokenIdTensor.setData(tokenIdBufffer);
        inputIdTensor.setData(inputIdBufffer);
        maskIdTensor.setData(maskIdBufffer);
        return labels;
    }

    @Override
    public int padSamples() {
        if (batchSize <= 0) {
            logger.severe(Common.addTag("batch size should bigger than 0"));
            return -1;
        }
        logger.info(Common.addTag("before pad samples size:" + features.size()));
        int curSize = features.size();
        int modSize = curSize - curSize / batchSize * batchSize;
        padSize = modSize != 0 ? batchSize - modSize : 0;
        for (int i = 0; i < padSize; i++) {
            int idx = (int) (Math.random() * curSize);
            features.add(features.get(idx));
        }
        trainSampleSize = features.size();
        batchNum = features.size() / batchSize;
        logger.info(Common.addTag("after pad samples size:" + features.size()));
        logger.info(Common.addTag("after pad batch num:" + batchNum));
        return 0;
    }
}
