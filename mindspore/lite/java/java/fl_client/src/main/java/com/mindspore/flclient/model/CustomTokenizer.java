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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * custom tokenizer class
 *
 * @since v1.0
 */
public class CustomTokenizer {
    private static final Logger logger = Logger.getLogger(CustomTokenizer.class.toString());
    private Map<String, Integer> vocabs = new HashMap<>();
    private final int maxInputChars = 100;
    private final String[] notSplitStrs = {"UNK"};
    private int maxSeqLen = 8;
    private Map<String, Integer> labelMap = new HashMap<String, Integer>() {
        {
            put("good", 0);
            put("leimu", 1);
            put("xiaoku", 2);
            put("xin", 3);
            put("other", 4);
        }
    };

    private List<String> getPieceToken(String token) {
        List<String> subTokens = new ArrayList<>();
        boolean isBad = false;
        int start = 0;
        int tokenLen = token.length();
        while (start < tokenLen) {
            int end = tokenLen;
            String curStr = "";
            while (start < end) {
                String subStr = token.substring(start, end);
                if (start > 0) {
                    subStr = "##" + subStr;
                }
                if (vocabs.get(subStr) != null) {
                    curStr = subStr;
                    break;
                }
                end = end - 1;
            }
            if (curStr.isEmpty()) {
                isBad = true;
                break;
            }
            subTokens.add(curStr);
            start = end;
        }
        if (isBad) {
            return new ArrayList<>(Collections.singletonList("[UNK]"));
        } else {
            return subTokens;
        }
    }

    /**
     * init tokenizer
     *
     * @param vocabFile vocab file path
     * @param idsFile word id file path
     * @param seqLen max word len to clamp
     */
    public void init(String vocabFile, String idsFile, int seqLen) {
        if (vocabFile == null || idsFile == null) {
            logger.severe(Common.addTag("idsFile,vocabFile cannot be empty"));
            return;
        }
        Path vocabPath = Paths.get(vocabFile);
        List<String> vocabLines;
        try {
            vocabLines = Files.readAllLines(vocabPath, StandardCharsets.UTF_8);
        } catch (IOException e) {
            logger.severe(Common.addTag("read vocab file failed, please check vocab file path"));
            return;
        }
        Path idsPath = Paths.get(idsFile);
        List<String> idsLines;
        try {
            idsLines = Files.readAllLines(idsPath, StandardCharsets.UTF_8);
        } catch (IOException e) {
            logger.severe(Common.addTag("read ids file failed, please check ids file path"));
            return;
        }
        for (int i = 0; i < idsLines.size(); ++i) {
            try {
                vocabs.put(vocabLines.get(i), Integer.parseInt(idsLines.get(i)));
            } catch (NumberFormatException e) {
                logger.severe(Common.addTag("id lines has invalid content"));
                return;
            }
        }
        maxSeqLen = seqLen;
    }

    /**
     * check char is chinese char or punc char
     *
     * @param trimChar input char
     * @return ture if char is chinese char or punc char,else false
     */
    public Boolean isChineseOrPunc(char trimChar) {
        // is chinese char
        if (trimChar >= '\u4e00' && trimChar <= '\u9fa5') {
            return true;
        }
        // is puncuation char
        boolean isFrontPuncChar = (trimChar >= 33 && trimChar <= 47) || (trimChar >= 58 && trimChar <= 64);
        boolean isBackPuncChar = (trimChar >= 91 && trimChar <= 96) || (trimChar >= 123 && trimChar <= 126);
        return isFrontPuncChar || isBackPuncChar;
    }

    /**
     * split text
     *
     * @param text input text
     * @return split string array
     */
    public String[] splitText(String text) {
        if (text == null) {
            return new String[0];
        }
        // clean remove white and control char
        String trimText = text.trim();
        StringBuilder cleanText = new StringBuilder();
        for (int i = 0; i < trimText.length(); i++) {
            if (isChineseOrPunc(trimText.charAt(i))) {
                cleanText.append(" ").append(trimText.charAt(i)).append(" ");
            } else {
                cleanText.append(trimText.charAt(i));
            }
        }
        return cleanText.toString().trim().split("\\s+");
    }

    /**
     * combine token to piece
     *
     * @param tokens input tokens
     * @return pieces
     */
    public List<String> wordPieceTokenize(String[] tokens) {
        if (tokens == null) {
            return new ArrayList<>();
        }
        List<String> outputTokens = new ArrayList<>();
        for (String token : tokens) {
            List<String> subTokens = getPieceToken(token);
            outputTokens.addAll(subTokens);
        }
        return outputTokens;
    }

    /**
     * convert token to id
     *
     * @param tokens input tokens
     * @param isCycTrunc if need cyc trunc
     * @return ids
     */
    public List<Integer> convertTokensToIds(List<String> tokens, boolean isCycTrunc) {
        int seqLen = tokens.size();
        List<String> truncTokens;
        if (tokens.size() > maxSeqLen - 2) {
            if (isCycTrunc) {
                int randIndex = (int) (Math.random() * seqLen);
                if (randIndex > seqLen - maxSeqLen + 2) {
                    List<String> rearPart = tokens.subList(randIndex, seqLen);
                    List<String> frontPart = tokens.subList(0, randIndex + maxSeqLen - 2 - seqLen);
                    rearPart.addAll(frontPart);
                    truncTokens = rearPart;
                } else {
                    truncTokens = tokens.subList(randIndex, randIndex + maxSeqLen - 2);
                }
            } else {
                truncTokens = tokens.subList(0, maxSeqLen - 2);
            }
        } else {
            truncTokens = new ArrayList<>(tokens);
        }
        truncTokens.add(0, "[CLS]");
        truncTokens.add("[SEP]");
        List<Integer> ids = new ArrayList<>(truncTokens.size());
        for (String token : truncTokens) {
            ids.add(vocabs.getOrDefault(token, vocabs.get("[UNK]")));
        }
        return ids;
    }

    /**
     * add random mask and replace feature
     *
     * @param feature text feature
     * @param isKeptFirst if keep first char not change
     * @param isKeptLast if keep last char not change
     */
    public void addRandomMaskAndReplace(Feature feature, boolean isKeptFirst, boolean isKeptLast) {
        if (feature == null) {
            return;
        }
        int[] masks = new int[maxSeqLen];
        Arrays.fill(masks, 1);
        int[] replaces = new int[maxSeqLen];
        Arrays.fill(replaces, 1);
        int[] inputIds = feature.inputIds;
        for (int i = 0; i < feature.seqLen; i++) {
            double rand1 = Math.random();
            if (rand1 < 0.15) {
                masks[i] = 0;
                double rand2 = Math.random();
                if (rand2 < 0.8) {
                    replaces[i] = 103;
                } else if (rand2 < 0.9) {
                    masks[i] = 1;
                } else {
                    int vocabSize = 11682;
                    replaces[i] = (int) (Math.random() * vocabSize);
                }
            }
            if (isKeptFirst) {
                masks[i] = 1;
                replaces[i] = 0;
            }
            if (isKeptLast) {
                masks[feature.seqLen - 1] = 1;
                replaces[feature.seqLen - 1] = 0;
            }
            inputIds[i] = inputIds[i] * masks[i] + replaces[i];
        }
    }

    /**
     * get feature
     *
     * @param tokens input tokens
     * @param label input label
     * @return feature
     */
    public Optional<Feature> getFeatures(List<Integer> tokens, String label) {
        if (tokens == null || label == null) {
            logger.warning(Common.addTag("tokens or label is null"));
            return Optional.empty();
        }
        if (!labelMap.containsKey(label)) {
            return Optional.empty();
        }
        int[] segmentIds = new int[maxSeqLen];
        Arrays.fill(segmentIds, 0);
        int[] masks = new int[maxSeqLen];
        Arrays.fill(masks, 0);
        Arrays.fill(masks, 0, tokens.size(), 1); // tokens size can ensure less than masks
        int[] inputIds = new int[maxSeqLen];
        Arrays.fill(inputIds, 0);
        for (int i = 0; i < tokens.size(); i++) {
            inputIds[i] = tokens.get(i);
        }
        return Optional.of(new Feature(inputIds, masks, segmentIds, labelMap.get(label), tokens.size()));
    }

    /**
     * tokenize text to tokens
     *
     * @param text input tokens
     * @param isTrainMode if work in train mod
     * @return tokens
     */
    public List<Integer> tokenize(String text, boolean isTrainMode) {
        if (text == null) {
            logger.warning(Common.addTag("text is empty,skip it"));
            return new ArrayList<>();
        }
        String[] splitTokens = splitText(text);
        List<String> wordPieceTokens = wordPieceTokenize(splitTokens);
        return convertTokensToIds(wordPieceTokens, isTrainMode); // trainMod need cyclicTrunc
    }
}

