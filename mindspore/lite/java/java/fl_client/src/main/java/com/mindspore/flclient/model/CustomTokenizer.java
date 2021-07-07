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
import java.util.*;
import java.util.logging.Logger;

public class CustomTokenizer {
    private static final Logger logger = Logger.getLogger(CustomTokenizer.class.toString());
    private Map<String, Integer> vocabs = new HashMap<>();
    private Boolean doLowerCase = Boolean.TRUE;
    private int maxInputChars = 100;
    private String[] NotSplitStrs = {"UNK"};
    private String unkToken = "[UNK]";
    private int maxSeqLen = 8;
    private int vocabSize = 11682;
    private Map<String, Integer> labelMap = new HashMap<String, Integer>() {{
        put("good", 0);
        put("leimu", 1);
        put("xiaoku", 2);
        put("xin", 3);
        put("other", 4);
    }};

    public void init(String vocabFile, String idsFile, boolean trainMod, boolean doLowerCase) {
        this.doLowerCase = doLowerCase;
        Path vocabPath = Paths.get(vocabFile);
        List<String> vocabLines = null;
        try {
            vocabLines = Files.readAllLines(vocabPath, StandardCharsets.UTF_8);
        } catch (IOException e) {
            logger.severe(Common.addTag("read vocab file failed," + e.getMessage()));
        }
        if (vocabLines == null) {
            logger.severe(Common.addTag("vocabLines cannot be null"));
            return;
        }
        Path idsPath = Paths.get(idsFile);
        List<String> idsLines = null;
        try {
            idsLines = Files.readAllLines(idsPath, StandardCharsets.UTF_8);
        } catch (IOException e) {
            logger.severe(Common.addTag("read ids file failed," + e.getMessage()));
        }
        if (idsLines == null) {
            logger.severe(Common.addTag("idsLines cannot be null"));
            return;
        }
        for (int i = 0; i < idsLines.size(); ++i) {
            vocabs.put(vocabLines.get(i), Integer.parseInt(idsLines.get(i)));
        }
        if (!trainMod) {
            maxSeqLen = 256;
        }
    }

    // is chinses or punctuation
    public Boolean isChineseOrPunc(char trimChar) {
        // is chinese char
        if (trimChar >= '\u4e00' && trimChar <= '\u9fa5') {
            return true;
        }
        // is puncuation char
        return (trimChar >= 33 && trimChar <= 47) || (trimChar >= 58 && trimChar <= 64) || (trimChar >= 91 && trimChar
                <= 96) || (trimChar >= 123 && trimChar <= 126);
    }

    public String[] splitText(String text) {
        if (text.isEmpty()) {
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

    //   input = "unaffable" , output = ["un", "##aff", "##able"]
    public List<String> wordPieceTokenize(String[] tokens) {
        List<String> outputTokens = new ArrayList<>();
        for (String token : tokens) {
            List<String> subTokens = new ArrayList<>();
            boolean isBad = false;
            int start = 0;
            while (start < token.length()) {
                int end = token.length();
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
                outputTokens.add(unkToken);
            } else {
                outputTokens.addAll(subTokens);
            }
        }
        return outputTokens;

    }

    public List<Integer> convertTokensToIds(List<String> tokens, boolean cycTrunc) {
        int seqLen = tokens.size();
        if (tokens.size() > maxSeqLen - 2) {
            if (cycTrunc) {
                int randIndex = (int) (Math.random() * seqLen);
                if (randIndex > seqLen - maxSeqLen + 2) {
                    List<String> rearPart = tokens.subList(randIndex, seqLen);
                    List<String> frontPart = tokens.subList(0, randIndex + maxSeqLen - 2 - seqLen);
                    rearPart.addAll(frontPart);
                    tokens = rearPart;
                } else {
                    tokens = tokens.subList(randIndex, randIndex + maxSeqLen - 2);
                }
            } else {
                tokens = tokens.subList(0, maxSeqLen - 2);
            }
        }
        tokens.add(0, "[CLS]");
        tokens.add("[SEP]");
        List<Integer> ids = new ArrayList<>(tokens.size());
        for (String token : tokens) {
            ids.add(vocabs.getOrDefault(token, vocabs.get("[UNK]")));
        }
        return ids;
    }

    public void addRandomMaskAndReplace(Feature feature, boolean keepFirstUnchange, boolean keepLastUnchange) {
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
                    replaces[i] = (int) (Math.random() * vocabSize);
                }
            }
            if (keepFirstUnchange) {
                masks[i] = 1;
                replaces[i] = 0;
            }
            if (keepLastUnchange) {
                masks[feature.seqLen - 1] = 1;
                replaces[feature.seqLen - 1] = 0;
            }
            inputIds[i] = inputIds[i] * masks[i] + replaces[i];
        }
    }

    public Feature getFeatures(List<Integer> tokens, String label) {
        if (!labelMap.containsKey(label)) {
            logger.severe(Common.addTag("label map not contains label:" + label));
            return null;
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
        return new Feature(inputIds, masks, segmentIds, labelMap.get(label), tokens.size());
    }

    public List<Integer> tokenize(String text, boolean trainMod) {
        String[] splitTokens = splitText(text);
        List<String> wordPieceTokens = wordPieceTokenize(splitTokens);
        return convertTokensToIds(wordPieceTokens, trainMod); // trainMod need cyclicTrunc
    }
}

