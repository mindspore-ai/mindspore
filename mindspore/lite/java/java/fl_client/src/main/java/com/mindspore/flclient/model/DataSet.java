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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

public class DataSet {
    private static final Logger logger = Logger.getLogger(DataSet.class.toString());

    public static List<Feature> init(String trainFile, String vocabFile, String idsFile, boolean trainMod) {
        if (trainFile.isEmpty() || vocabFile.isEmpty() || idsFile.isEmpty()) {
            logger.severe(Common.addTag("dataset init failed,trainFile,idsFile,vocabFile cannot be empty"));
            return null;
        }
        // read train file
        CustomTokenizer customTokenizer = new CustomTokenizer();
        customTokenizer.init(vocabFile, idsFile, trainMod, true);
        List<String> allLines = readTxtFile(trainFile);
        if (allLines == null) {
            logger.severe(Common.addTag("all lines cannot be null"));
            return null;
        }
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
            List<Integer> tokens = customTokenizer.tokenize(examples.get(i), trainMod);
            Feature feature = customTokenizer.getFeatures(tokens, labels.get(i));
            if (trainMod) {
                customTokenizer.addRandomMaskAndReplace(feature, true, true);
            }
            features.add(feature);
        }
        return features;
    }

    public static List<Feature> readInferData(String inferFile, String vocabFile, String idsFile, boolean trainMod) {
        if (inferFile.isEmpty() || vocabFile.isEmpty() || idsFile.isEmpty()) {
            logger.severe(Common.addTag("dataset init failed,trainFile,idsFile,vocabFile cannot be empty"));
            return null;
        }
        // read train file
        CustomTokenizer customTokenizer = new CustomTokenizer();
        customTokenizer.init(vocabFile, idsFile, false, true);
        List<String> allLines = readTxtFile(inferFile);
        if (allLines == null) {
            logger.severe(Common.addTag("all lines cannot be null"));
            return null;
        }
        List<Feature> features = new ArrayList<>(allLines.size());
        for (String line : allLines) {
            if (line.isEmpty()) {
                continue;
            }
            List<Integer> tokens = customTokenizer.tokenize(line, trainMod);
            Feature feature = customTokenizer.getFeatures(tokens, "other");
            features.add(feature);
        }
        return features;
    }

    public static byte[] readBinFile(String dataFile) {
        // read train file
        Path path = Paths.get(dataFile);
        byte[] data = null;
        try {
            data = Files.readAllBytes(path);
        } catch (IOException e) {
            logger.severe(Common.addTag("read ids file failed," + e.getMessage()));
        }
        return data;
    }

    private static List<String> readTxtFile(String file) {
        Path path = Paths.get(file);
        List<String> allLines = null;
        try {
            allLines = Files.readAllLines(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            logger.severe(Common.addTag("read file failed," + e.getMessage()));
        }
        return allLines;
    }
}

