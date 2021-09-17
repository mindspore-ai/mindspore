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
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * lenet for train
 *
 * @since v1.0
 */
public class TrainLenet extends TrainModel {
    private static final Logger logger = Logger.getLogger(TrainLenet.class.toString());

    private static final int NUM_OF_CLASS = 62;

    private static final int LENET_INPUT_SIZE = 2;

    private static volatile TrainLenet trainLenet;

    private int imageSize;

    private int labelSize;

    private byte[] imageArray;

    private int[] labelArray;

    private ByteBuffer labelIdBuffer;

    private ByteBuffer imageBuffer;

    private TrainLenet() {
    }

    /**
     * get singleton instance
     *
     * @return singleton instance
     */
    public static TrainLenet getInstance() {
        TrainLenet localRef = trainLenet;
        if (localRef == null) {
            synchronized (TrainLenet.class) {
                localRef = trainLenet;
                if (localRef == null) {
                    trainLenet = localRef = new TrainLenet();
                }
            }
        }
        return localRef;
    }

    /**
     * infer model
     *
     * @param modelPath model file path
     * @param imageFile image file path
     * @return infer result
     */
    public int[] inferModel(String modelPath, String imageFile) {
        if (modelPath == null || imageFile == null || modelPath.isEmpty() || imageFile.isEmpty()) {
            logger.severe(Common.addTag("model path or image file cannot be empty"));
            return new int[0];
        }
        // cur dont care return value,train sample size need sure after pad sample
        initDataSet(imageFile, "");
        int status = initSessionAndInputs(modelPath, false);
        if (status == -1) {
            logger.severe(Common.addTag("init session and inputs failed"));
            return new int[0];
        }
        status = padSamples();
        if (status == -1) {
            logger.severe(Common.addTag("infer model failed"));
            return new int[0];
        }
        int[] predictLabels = new int[trainSampleSize];
        for (int j = 0; j < batchNum; j++) {
            List<Integer> modelInput = fillModelInput(j, false);
            if (!modelInput.isEmpty()) {
                logger.severe(Common.addTag("infer mode model input need empty"));
                return new int[0];
            }
            boolean isSuccess = trainSession.runGraph();
            if (!isSuccess) {
                logger.severe(Common.addTag("run graph failed"));
                return new int[0];
            }
            int[] batchLabels = getBatchLabel();
            System.arraycopy(batchLabels, 0, predictLabels, j * batchSize, batchSize);
        }
        if (predictLabels.length == 0) {
            return new int[0];
        }
        return Arrays.copyOfRange(predictLabels, 0, trainSampleSize - padSize);
    }

    /**
     * init data set
     *
     * @param imageFile image file path
     * @param labelFile label file path
     * @return data set size
     */
    public int initDataSet(String imageFile, String labelFile) {
        if (imageFile != null && !imageFile.isEmpty()) {
            imageArray = DataSet.readBinFile(imageFile);
        }
        // train mod
        if (labelFile != null && !labelFile.isEmpty()) {
            byte[] labelByteArray = DataSet.readBinFile(labelFile);
            int trainSize = labelByteArray.length;
            trainSampleSize = trainSize / Integer.BYTES;
            // label is 32,need pad 32*62
            labelArray = new int[labelByteArray.length / 4 * NUM_OF_CLASS];
            Arrays.fill(labelArray, 0);
            int offset = 0;
            for (int i = 0; i < labelByteArray.length; i += 4) {
                labelArray[offset * NUM_OF_CLASS + labelByteArray[i]] = 1;
                offset++;
            }
        } else {
            labelArray = null;  // labelArray may be initialized from train
        }

        return trainSampleSize;
    }

    @Override
    public int padSamples() {
        if (batchSize <= 0) {
            logger.severe(Common.addTag("pad samples failed"));
            return -1;
        }
        if (labelArray == null) // infer model
        {
            labelArray = new int[imageArray.length * NUM_OF_CLASS / (imageSize / batchSize * Float.BYTES)];
            Arrays.fill(labelArray, 0);
        }
        int curSize = labelArray.length / NUM_OF_CLASS;
        int modSize = curSize - curSize / batchSize * batchSize;
        padSize = modSize != 0 ? batchSize * NUM_OF_CLASS - modSize : 0;
        if (padSize != 0) {
            int[] padLabelArray = new int[labelArray.length + padSize * numOfClass];
            int batchImageSize = imageSize / batchSize;
            byte[] padImageArray = new byte[imageArray.length + padSize * batchImageSize * Float.BYTES];
            System.arraycopy(labelArray, 0, padLabelArray, 0, labelArray.length);
            System.arraycopy(imageArray, 0, padImageArray, 0, imageArray.length);
            for (int i = 0; i < padSize; i++) {
                int idx = (int) (Math.random() * curSize);
                System.arraycopy(labelArray, idx * numOfClass, padLabelArray, labelArray.length + i * numOfClass,
                        numOfClass);
                System.arraycopy(imageArray, idx * batchImageSize * Float.BYTES, padImageArray,
                        padImageArray.length + i * batchImageSize * Float.BYTES, batchImageSize * Float.BYTES);
            }
            labelArray = padLabelArray;
            imageArray = padImageArray;
        }
        trainSampleSize = curSize + padSize;
        batchNum = trainSampleSize / batchSize;
        logger.info(Common.addTag("total samples:" + trainSampleSize));
        logger.info(Common.addTag("total batchNum:" + batchNum));
        return 0;
    }

    @Override
    public int initSessionAndInputs(String modelPath, boolean isTrainMode) {
        if (modelPath == null) {
            logger.severe(Common.addTag("modelPath cannot be empty"));
            return -1;
        }
        Optional<LiteSession> optTrainSession = SessionUtil.initSession(modelPath);
        if (!optTrainSession.isPresent()) {
            logger.severe(Common.addTag("session init failed"));
            return -1;
        }
        trainSession = optTrainSession.get();
        List<MSTensor> inputs = trainSession.getInputs();
        numOfClass = NUM_OF_CLASS;
        if (inputs.size() != LENET_INPUT_SIZE) {
            logger.severe(Common.addTag("lenet input size error"));
            return -1;
        }
        MSTensor imageTensor = inputs.get(0);
        if (imageTensor == null || imageTensor.getShape().length == 0) {
            logger.severe(Common.addTag("image tensor cannot be empty"));
            return -1;
        }
        batchSize = imageTensor.getShape()[0];
        imageSize = imageTensor.elementsNum();
        imageBuffer = ByteBuffer.allocateDirect(imageSize * Float.BYTES);
        imageBuffer.order(ByteOrder.nativeOrder());
        MSTensor labelTensor = inputs.get(1);
        if (labelTensor == null) {
            logger.severe(Common.addTag("labelTensor tensor cannot be empty"));
            return -1;
        }
        labelSize = labelTensor.elementsNum();
        labelIdBuffer = ByteBuffer.allocateDirect(labelSize * Integer.BYTES);
        labelIdBuffer.order(ByteOrder.nativeOrder());
        logger.info(Common.addTag("init session and inputs success"));
        return 0;
    }

    @Override
    public List<Integer> fillModelInput(int batchIdx, boolean isTrainMode) {
        imageBuffer.clear();
        labelIdBuffer.clear();
        if ((batchIdx + 1) * imageSize * Float.BYTES - 1 >= imageArray.length) {
            logger.severe(Common.addTag("fill model image input failed"));
            return new ArrayList<>();
        }
        List<Integer> predictLabels = new ArrayList<>(batchSize);
        for (int i = 0; i < imageSize * Float.BYTES; i++) {
            imageBuffer.put(imageArray[batchIdx * imageSize * Float.BYTES + i]);
        }
        if (labelArray == null || (batchIdx + 1) * labelSize - 1 >= labelArray.length) {
            logger.severe(Common.addTag("fill model label input failed"));
            return new ArrayList<>();
        }
        for (int i = 0; i < labelSize; i++) {
            labelIdBuffer.putFloat(labelArray[batchIdx * labelSize + i]);
            if (!isTrainMode && labelArray[batchIdx * labelSize + i] == 1) {
                predictLabels.add(i % NUM_OF_CLASS);
            }
        }

        List<MSTensor> inputs = trainSession.getInputs();
        if (inputs.size() != LENET_INPUT_SIZE) {
            logger.severe(Common.addTag("input size error"));
            return new ArrayList<>();
        }
        MSTensor imageTensor = inputs.get(0);
        MSTensor labelTensor = inputs.get(1);
        imageTensor.setData(imageBuffer);
        labelTensor.setData(labelIdBuffer);
        return predictLabels;
    }
}
