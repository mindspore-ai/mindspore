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

package com.mindspore.flclient.demo.albert;

import com.mindspore.flclient.model.DataSet;
import com.mindspore.flclient.model.RunType;
import com.mindspore.flclient.model.Status;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * Defining dataset for albert.
 *
 * @since v1.0
 */
public class AlbertDataSet extends DataSet {
    private static final Logger LOGGER = Logger.getLogger(AlbertDataSet.class.toString());
    private static final int INPUT_FILE_NUM = 3;
    private static final int IDS_FILE_INDEX = 2;
    private static final int WORD_SPLIT_NUM = 2;
    private static final int ALBERT_INPUT_SIZE = 4;
    private static final int MASK_INPUT_INDEX = 2;
    private static final int LABEL_INPUT_INDEX = 3;
    private final RunType runType;
    private final List<Feature> features = new ArrayList<>();
    private final int maxSeqLen;
    private final List<Integer> targetLabels = new ArrayList<>();

    /**
     * Defining a constructor of albert dataset.
     */
    public AlbertDataSet(RunType runType, int maxSeqLen) {
        this.runType = runType;
        this.maxSeqLen = maxSeqLen;
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
        if (inputsBuffer.size() != ALBERT_INPUT_SIZE) {
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
        ByteBuffer tokenIdBuffer = inputsBuffer.get(0);
        ByteBuffer inputIdBufffer = inputsBuffer.get(1);
        ByteBuffer maskIdBufffer = inputsBuffer.get(MASK_INPUT_INDEX);
        ByteBuffer labelIdBufffer = inputsBuffer.get(LABEL_INPUT_INDEX);
        for (int i = 0; i < batchSize; i++) {
            Feature feature = features.get(batchIdx * batchSize + i);
            for (int j = 0; j < maxSeqLen; j++) {
                inputIdBufffer.putInt(feature.inputIds[j]);
                tokenIdBuffer.putInt(feature.tokenIds[j]);
                maskIdBufffer.putInt(feature.inputMasks[j]);
            }
            if (runType != RunType.INFERMODE) {
                labelIdBufffer.putInt(feature.labelIds);
                targetLabels.add(feature.labelIds);
            }
        }
    }

    @Override
    public void shuffle() {
    }

    @Override
    public void padding() {
        if (batchSize <= 0) {
            LOGGER.severe("batch size should bigger than 0");
            return;
        }
        LOGGER.info("before pad samples size:" + features.size());
        int curSize = features.size();
        int modSize = curSize - curSize / batchSize * batchSize;
        int padSize = modSize != 0 ? batchSize - modSize : 0;
        for (int i = 0; i < padSize; i++) {
            int idx = (int) (Math.random() * curSize);
            features.add(features.get(idx));
        }
        sampleSize = features.size();
        batchNum = features.size() / batchSize;
        LOGGER.info("after pad samples size:" + features.size());
        LOGGER.info("after pad batch num:" + batchNum);
    }

    private static List<String> readTxtFile(String file) {
        if (file == null) {
            LOGGER.severe("file cannot be empty");
            return new ArrayList<>();
        }
        Path path = Paths.get(file);
        List<String> allLines = new ArrayList<>();
        try {
            allLines = Files.readAllLines(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            LOGGER.severe("read txt file failed,please check txt file path");
        }
        return allLines;
    }

    private Status ConvertTrainData(String trainFile, String vocabFile, String idsFile) {
        if (trainFile == null || vocabFile == null || idsFile == null) {
            LOGGER.severe("dataset init failed,trainFile,idsFile,vocabFile cannot be empty");
            return Status.NULLPTR;
        }
        // read train file
        CustomTokenizer customTokenizer = new CustomTokenizer();
        customTokenizer.init(vocabFile, idsFile, maxSeqLen);
        List<String> allLines = readTxtFile(trainFile);
        List<String> examples = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        for (String line : allLines) {
            String[] tokens = line.split(">>>");
            if (tokens.length != WORD_SPLIT_NUM) {
                LOGGER.warning("line may have format problem,need include >>>");
                continue;
            }
            examples.add(tokens[1]);
            tokens = tokens[0].split("<<<");
            if (tokens.length != WORD_SPLIT_NUM) {
                LOGGER.warning("line may have format problem,need include >>>");
                continue;
            }
            labels.add(tokens[1]);
        }

        for (int i = 0; i < examples.size(); i++) {
            List<Integer> tokens = customTokenizer.tokenize(examples.get(i), runType == RunType.TRAINMODE);
            if (tokens.isEmpty()) {
                continue;
            }
            Optional<Feature> feature = customTokenizer.getFeatures(tokens, labels.get(i));
            if (!feature.isPresent()) {
                continue;
            }
            if (runType == RunType.TRAINMODE) {
                customTokenizer.addRandomMaskAndReplace(feature.get(), true, true);
            }
            feature.ifPresent(features::add);
        }
        sampleSize = features.size();
        return Status.SUCCESS;
    }

    private Status ConvertInferData(String inferFile, String vocabFile, String idsFile) {
        if (inferFile == null || vocabFile == null || idsFile == null) {
            LOGGER.severe("dataset init failed,trainFile,idsFile,vocabFile cannot be empty");
            return Status.NULLPTR;
        }
        // read train file
        CustomTokenizer customTokenizer = new CustomTokenizer();
        customTokenizer.init(vocabFile, idsFile, maxSeqLen);
        List<String> allLines = readTxtFile(inferFile);
        for (String line : allLines) {
            if (line.isEmpty()) {
                continue;
            }
            List<Integer> tokens = customTokenizer.tokenize(line, runType == RunType.TRAINMODE);
            Optional<Feature> feature = customTokenizer.getFeatures(tokens, "other");
            if (!feature.isPresent()) {
                continue;
            }
            features.add(feature.get());
        }
        sampleSize = features.size();
        return Status.SUCCESS;
    }

    @Override
    public Status dataPreprocess(List<String> files) {
        if (files.size() != INPUT_FILE_NUM) {
            LOGGER.severe("files size error");
            return Status.FAILED;
        }
        String dataFile = files.get(0);
        String vocabFile = files.get(1);
        String idsFile = files.get(IDS_FILE_INDEX);
        if (runType == RunType.TRAINMODE || runType == RunType.EVALMODE) {
            return ConvertTrainData(dataFile, vocabFile, idsFile);
        } else {
            return ConvertInferData(dataFile, vocabFile, idsFile);
        }
    }
}
