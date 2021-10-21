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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * dataset class
 *
 * @since v1.0
 */
public class DataSet {
    private static final Logger logger = Logger.getLogger(DataSet.class.toString());

    /**
     * init dataset
     *
     * @param trainFile   train file path
     * @param vocabFile   vocab file path
     * @param idsFile     id file path
     * @param isTrainMode train mod
     * @param maxSeqLen   max seq len to clamp
     * @return features
     */
    public static List<Feature> init(String trainFile, String vocabFile, String idsFile, boolean isTrainMode,
                                     int maxSeqLen) {
        if (trainFile == null || vocabFile == null || idsFile == null) {
            logger.severe(Common.addTag("dataset init failed,trainFile,idsFile,vocabFile cannot be empty"));
            return new ArrayList<>();
        }
        // read train file
        CustomTokenizer customTokenizer = new CustomTokenizer();
        customTokenizer.init(vocabFile, idsFile, maxSeqLen);
        List<String> allLines = readTxtFile(trainFile);
        List<String> examples = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        for (String line : allLines) {
            String[] tokens = line.split(">>>");
            if (tokens.length != 2) {
                logger.warning(Common.addTag("line may have format problem,need include >>>"));
                continue;
            }
            examples.add(tokens[1]);
            tokens = tokens[0].split("<<<");
            if (tokens.length != 2) {
                logger.warning(Common.addTag("line may have format problem,need include >>>"));
                continue;
            }
            labels.add(tokens[1]);
        }

        List<Feature> features = new ArrayList<>(examples.size());
        for (int i = 0; i < examples.size(); i++) {
            List<Integer> tokens = customTokenizer.tokenize(examples.get(i), isTrainMode);
            if (tokens.isEmpty()) {
                continue;
            }
            Optional<Feature> feature = customTokenizer.getFeatures(tokens, labels.get(i));
            if (isTrainMode) {
                customTokenizer.addRandomMaskAndReplace(feature.get(), true, true);
            }
            features.add(feature.get());
        }
        return features;
    }

    /**
     * read infer data
     *
     * @param inferFile infer file path
     * @param vocabFile vocab file path
     * @param idsFile ids file path
     * @param isTrainMode if in train mod
     * @return infer features
     */
    public static List<Feature> readInferData(String inferFile, String vocabFile, String idsFile, boolean isTrainMode
            , int maxSeqLen) {
        if (inferFile == null || vocabFile == null || idsFile == null) {
            logger.severe(Common.addTag("dataset init failed,trainFile,idsFile,vocabFile cannot be empty"));
            return new ArrayList<>();
        }
        // read train file
        CustomTokenizer customTokenizer = new CustomTokenizer();
        customTokenizer.init(vocabFile, idsFile, maxSeqLen);
        List<String> allLines = readTxtFile(inferFile);
        List<Feature> features = new ArrayList<>(allLines.size());
        for (String line : allLines) {
            if (line.isEmpty()) {
                continue;
            }
            List<Integer> tokens = customTokenizer.tokenize(line, isTrainMode);
            Optional<Feature> feature = customTokenizer.getFeatures(tokens, "other");
            if (!feature.isPresent()) {
                continue;
            }
            features.add(feature.get());
        }
        return features;
    }

    /**
     * read bin file
     *
     * @param dataFile data file
     * @return data array
     */
    public static byte[] readBinFile(String dataFile) {
        if (dataFile == null || dataFile.isEmpty()) {
            logger.severe(Common.addTag("file cannot be empty"));
            return new byte[0];
        }
        // read train file
        Path path = Paths.get(dataFile);
        byte[] data = new byte[0];
        try {
            data = Files.readAllBytes(path);
        } catch (IOException e) {
            logger.severe(Common.addTag("read data file failed,please check data file path"));
        }
        return data;
    }

    private static List<String> readTxtFile(String file) {
        if (file == null) {
            logger.severe(Common.addTag("file cannot be empty"));
            return new ArrayList<>();
        }
        Path path = Paths.get(file);
        List<String> allLines = new ArrayList<>();
        try {
            allLines = Files.readAllLines(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            logger.severe(Common.addTag("read txt file failed,please check txt file path"));
        }
        return allLines;
    }
}

