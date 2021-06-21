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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

public class AdBert extends TrainModel {
    private static final Logger logger = Logger.getLogger(AdBert.class.toString());

    private static final int NUM_OF_CLASS = 5;

    List<Feature> features;

    private int dataSize;

    private ByteBuffer inputIdBufffer;

    private ByteBuffer tokenIdBufffer;

    private ByteBuffer maskIdBufffer;

    private ByteBuffer labelIdBufffer;

    @Override
    public int initSessionAndInputs(String modelPath, boolean trainMod) {
        int ret = -1;
        trainSession = SessionUtil.initSession(modelPath);
        if (trainSession == null) {
            logger.severe(Common.addTag("session init failed"));
            return ret;
        }
        List<MSTensor> inputs = trainSession.getInputs();
        MSTensor labelIdTensor = inputs.get(0);
        int inputSize = labelIdTensor.elementsNum(); // labelId,tokenId,inputId,maskId has same size
        batchSize = labelIdTensor.getShape()[0];
        if (batchSize <= 0) {
            logger.severe(Common.addTag("batch size should bigger than 0"));
            return ret;
        }
        dataSize = inputSize / batchSize;
        inputIdBufffer = ByteBuffer.allocateDirect(inputSize * Integer.BYTES);
        tokenIdBufffer = ByteBuffer.allocateDirect(inputSize * Integer.BYTES);
        maskIdBufffer = ByteBuffer.allocateDirect(inputSize * Integer.BYTES);
        inputIdBufffer.order(ByteOrder.nativeOrder());
        tokenIdBufffer.order(ByteOrder.nativeOrder());
        maskIdBufffer.order(ByteOrder.nativeOrder());
        if (trainMod) {
            labelIdBufffer = ByteBuffer.allocateDirect(inputSize * Integer.BYTES);
            labelIdBufffer.order(ByteOrder.nativeOrder());
        }
        numOfClass = NUM_OF_CLASS;
        return 0;
    }

    @Override
    public List<Integer> fillModelInput(int batchIdx, boolean trainMod) {
        inputIdBufffer.clear();
        tokenIdBufffer.clear();
        maskIdBufffer.clear();
        if (trainMod) {
            labelIdBufffer.clear();
        }
        List<Integer> labels = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            Feature feature = features.get(batchIdx * batchSize + i);
            for (int j = 0; j < dataSize; j++) {
                inputIdBufffer.putInt(feature.inputIds[j]);
            }
            for (int j = 0; j < dataSize; j++) {
                tokenIdBufffer.putInt(feature.tokenIds[j]);
            }
            for (int j = 0; j < dataSize; j++) {
                maskIdBufffer.putInt(feature.inputMasks[j]);
            }
            if (!trainMod) {
                labels.add(feature.labelIds);
            }
            if (trainMod) {
                for (int j = 0; j < dataSize; j++) {
                    labelIdBufffer.putInt(feature.inputIds[j]);
                }
            }
        }

        List<MSTensor> inputs = trainSession.getInputs();
        MSTensor labelIdTensor;
        MSTensor tokenIdTensor;
        MSTensor inputIdTensor;
        MSTensor maskIdTensor;
        if (trainMod) {
            labelIdTensor = inputs.get(0);
            tokenIdTensor = inputs.get(1);
            inputIdTensor = inputs.get(2);
            maskIdTensor = inputs.get(3);
            labelIdTensor.setData(labelIdBufffer);
        } else {
            tokenIdTensor = inputs.get(0);
            inputIdTensor = inputs.get(1);
            maskIdTensor = inputs.get(2);
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
        return 0;
    }
}
