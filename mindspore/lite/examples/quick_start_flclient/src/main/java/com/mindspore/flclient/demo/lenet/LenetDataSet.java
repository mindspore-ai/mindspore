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

package com.mindspore.flclient.demo.lenet;

import com.mindspore.flclient.model.DataSet;
import com.mindspore.flclient.model.Status;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * Defining the minist dataset for lenet.
 *
 * @since v1.0
 */
public class LenetDataSet extends DataSet {
    private static final Logger LOGGER = Logger.getLogger(LenetDataSet.class.toString());
    private static final int IMAGE_SIZE = 32 * 32 * 3;
    private static final int FLOAT_BYTE_SIZE = 4;
    private byte[] imageArray;
    private int[] labelArray;
    private final int numOfClass;
    private List<Integer> targetLabels;

    /**
     * Defining a constructor of lenet dataset.
     */
    public LenetDataSet(int numOfClass) {
        this.numOfClass = numOfClass;
    }

    /**
     * Get dataset labels.
     *
     * @return dataset target labels.
     */
    public List<Integer> getTargetLabels() {
        return targetLabels;
    }

    @Override
    public void fillInputBuffer(List<ByteBuffer> inputsBuffer, int batchIdx) {
        // infer,train,eval model is same one
        if (inputsBuffer.size() != 2) {
            LOGGER.severe("input size error");
            return;
        }
        if (batchIdx > batchNum) {
            LOGGER.severe("fill model image input failed");
            return;
        }
        for (ByteBuffer inputBuffer : inputsBuffer) {
            inputBuffer.clear();
        }
        ByteBuffer imageBuffer = inputsBuffer.get(0);
        ByteBuffer labelIdBuffer = inputsBuffer.get(1);
        int imageInputBytes = IMAGE_SIZE * batchSize * Float.BYTES;
        for (int i = 0; i < imageInputBytes; i++) {
            imageBuffer.put(imageArray[batchIdx * imageInputBytes + i]);
        }
        if (labelArray == null) {
            return;
        }
        int labelSize = batchSize * numOfClass;
        if ((batchIdx + 1) * labelSize - 1 >= labelArray.length) {
            LOGGER.severe("fill model label input failed");
            return;
        }
        labelIdBuffer.clear();
        for (int i = 0; i < labelSize; i++) {
            labelIdBuffer.putFloat(labelArray[batchIdx * labelSize + i]);
        }
    }

    @Override
    public void shuffle() {

    }

    @Override
    public void padding() {
        if (labelArray == null) // infer model
        {
            labelArray = new int[imageArray.length * numOfClass / (IMAGE_SIZE * Float.BYTES)];
            Arrays.fill(labelArray, 0);
        }
        int curSize = labelArray.length / numOfClass;
        int modSize = curSize - curSize / batchSize * batchSize;
        int padSize = modSize != 0 ? batchSize * numOfClass - modSize : 0;
        if (padSize != 0) {
            int[] padLabelArray = new int[labelArray.length + padSize * numOfClass];
            byte[] padImageArray = new byte[imageArray.length + padSize * IMAGE_SIZE * Float.BYTES];
            System.arraycopy(labelArray, 0, padLabelArray, 0, labelArray.length);
            System.arraycopy(imageArray, 0, padImageArray, 0, imageArray.length);
            for (int i = 0; i < padSize; i++) {
                int idx = (int) (Math.random() * curSize);
                System.arraycopy(labelArray, idx * numOfClass, padLabelArray, labelArray.length + i * numOfClass,
                        numOfClass);
                System.arraycopy(imageArray, idx * IMAGE_SIZE * Float.BYTES, padImageArray,
                        padImageArray.length + i * IMAGE_SIZE * Float.BYTES, IMAGE_SIZE * Float.BYTES);
            }
            labelArray = padLabelArray;
            imageArray = padImageArray;
        }
        sampleSize = curSize + padSize;
        batchNum = sampleSize / batchSize;
        setPredictLabels(labelArray);
        LOGGER.info("total samples:" + sampleSize);
        LOGGER.info("total batchNum:" + batchNum);
    }

    private void setPredictLabels(int[] labelArray) {
        int labels_num = labelArray.length / numOfClass;
        targetLabels = new ArrayList<>(labels_num);
        for (int i = 0; i < labels_num; i++) {
            int label = getMaxIndex(labelArray, numOfClass * i, numOfClass * (i + 1));
            if (label == -1) {
                LOGGER.severe("get max index failed");
            }
            targetLabels.add(label);
        }
    }

    private int getMaxIndex(int[] nums, int begin, int end) {
        for (int i = begin; i < end; i++) {
            if (nums[i] == 1) {
                return i - begin;
            }
        }
        return -1;
    }

    public static byte[] readBinFile(String dataFile) {
        if (dataFile == null || dataFile.isEmpty()) {
            LOGGER.severe("file cannot be empty");
            return new byte[0];
        }
        // read train file
        Path path = Paths.get(dataFile);
        byte[] data = new byte[0];
        try {
            data = Files.readAllBytes(path);
        } catch (IOException e) {
            LOGGER.severe("read data file failed,please check data file path");
        }
        return data;
    }

    @Override
    public Status dataPreprocess(List<String> files) {
        String labelFile = "";
        String imageFile;
        if (files.size() == 2) {
            imageFile = files.get(0);
            labelFile = files.get(1);
        } else if (files.size() == 1) {
            imageFile = files.get(0);
        } else {
            LOGGER.severe("files size error");
            return Status.FAILED;
        }
        imageArray = readBinFile(imageFile);
        if (labelFile != null && !labelFile.isEmpty()) {
            byte[] labelByteArray = readBinFile(labelFile);
            targetLabels = new ArrayList<>(labelByteArray.length / FLOAT_BYTE_SIZE);
            // model labels use one hot
            labelArray = new int[labelByteArray.length / FLOAT_BYTE_SIZE * numOfClass];
            Arrays.fill(labelArray, 0);
            int offset = 0;
            for (int i = 0; i < labelByteArray.length; i += FLOAT_BYTE_SIZE) {
                labelArray[offset * numOfClass + labelByteArray[i]] = 1;
                offset++;
            }
        } else {
            labelArray = null;  // labelArray may be initialized from train
        }
        sampleSize = imageArray.length / IMAGE_SIZE / FLOAT_BYTE_SIZE;
        return Status.SUCCESS;
    }
}
