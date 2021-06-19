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
package com.mindspore.flclient;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

public class Common {
    public static final String LOG_TITLE = "<FLClient> ";
    private static final Logger LOGGER = Logger.getLogger(Common.class.toString());
    private static List<String> flNameTrustList = new ArrayList<>(Arrays.asList("lenet", "adbert"));

    public static String generateUrl(boolean useElb, String ip, int port, int serverNum) {
        String url;
        if (useElb) {
            Random rand = new Random();
            int randomNum = rand.nextInt(100000) % serverNum + port;
            url = ip + String.valueOf(randomNum);
        } else {
            url = ip + String.valueOf(port);
        }
        return url;
    }

    public static void setClassifierWeightName(List<String> classifierWeightName) {
        classifierWeightName.add("albert.pooler.weight");
        classifierWeightName.add("albert.pooler.bias");
        classifierWeightName.add("classifier.weight");
        classifierWeightName.add("classifier.bias");
        LOGGER.info(addTag("classifierWeightName size: " + classifierWeightName.size()));
    }

    public static void setAlbertWeightName(List<String> albertWeightName) {
        albertWeightName.add("albert.encoder.embedding_hidden_mapping_in.weight");
        albertWeightName.add("albert.encoder.embedding_hidden_mapping_in.bias");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.query.weight");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.query.bias");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.key.weight");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.key.bias");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.value.weight");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention.value.bias");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.dense.weight");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.dense.bias");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.layernorm.gamma");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.attention.output.layernorm.beta");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.gamma");
        albertWeightName.add("albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.beta");
        LOGGER.info(addTag("albertWeightName size: " + albertWeightName.size()));
    }

    public static boolean checkFLName(String flName) {
        return (flNameTrustList.contains(flName));
    }

    public static void sleep(long millis) {
        try {
            Thread.sleep(millis);                 //1000 milliseconds is one second.
        } catch (InterruptedException ex) {
            LOGGER.severe(addTag("[sleep] catch InterruptedException: " + ex.getMessage()));
            Thread.currentThread().interrupt();
        }
    }

    public static long getWaitTime(String nextRequestTime) {

        Date date = new Date();
        long currentTime = date.getTime();
        long waitTime = 0;
        if (!nextRequestTime.equals("")) {
            waitTime = Math.max(0, Long.valueOf(nextRequestTime) - currentTime);
        }
        LOGGER.info(addTag("[getWaitTime] next request time stamp: " + nextRequestTime + " current time stamp: " + currentTime));
        LOGGER.info(addTag("[getWaitTime] waitTime: " + waitTime));
        return waitTime;
    }

    public static long startTime(String tag) {
        Date startDate = new Date();
        long startTime = startDate.getTime();
        LOGGER.info(addTag("[start time] <" + tag + "> start time: " + startTime));
        return startTime;
    }

    public static void endTime(long start, String tag) {
        Date endDate = new Date();
        long endTime = endDate.getTime();
        LOGGER.info(addTag("[end time] <" + tag + "> end time: " + endTime));
        LOGGER.info(addTag("[interval time] <" + tag + "> interval time(ms): " + (endTime - start)));
    }

    public static String addTag(String message) {
        return LOG_TITLE + message;
    }

    public static boolean isAutoscaling(byte[] message, String autoscalingTag) {
        return (new String(message)).contains(autoscalingTag);
    }

    public static boolean checkPath(String path) {
        File file = new File(path);
        return file.exists();
    }
}
