/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

import com.mindspore.flclient.model.AlInferBert;
import com.mindspore.flclient.model.AlTrainBert;
import com.mindspore.flclient.model.Client;
import com.mindspore.flclient.model.ClientManager;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.Status;
import com.mindspore.flclient.model.TrainLenet;
import mindspore.schema.ResponseCode;
import org.bouncycastle.crypto.BlockCipher;
import org.bouncycastle.crypto.engines.AESEngine;
import org.bouncycastle.crypto.prng.SP800SecureRandomBuilder;

import java.io.File;
import java.io.IOException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.mindspore.flclient.FLParameter.MAX_SLEEP_TIME;
import static com.mindspore.flclient.FLParameter.MAX_WAIT_TRY_TIME;
import static com.mindspore.flclient.FLParameter.SLEEP_TIME;
import static com.mindspore.flclient.LocalFLParameter.ALBERT;
import static com.mindspore.flclient.LocalFLParameter.LENET;

/**
 * Define basic global methods used in federated learning task.
 *
 * @since 2021-06-30
 */
public class Common {
    /**
     * Global logger title.
     */
    public static final String LOG_TITLE = "<FLClient> ";

    public static final String LOG_DEPRECATED = "This method will be deprecated in the next version, it is " +
            "recommended to " +
            "use the latest method according to the use cases in the official website tutorial";
    /**
     * The list of trust flName.
     */
    public static final List<String> FL_NAME_TRUST_LIST = new ArrayList<>(Arrays.asList("lenet", "albert"));

    /**
     * The list of trust ssl protocol.
     */
    public static final List<String> SSL_PROTOCOL_TRUST_LIST = new ArrayList<>(Arrays.asList("TLSv1.3", "TLSv1.2"));

    /**
     * The tag when server is in safe mode.
     */
    public static final String SAFE_MOD = "The cluster is in safemode.";

    /**
     * The tag when server is not ready.
     */
    public static final String JOB_NOT_AVAILABLE = "The server's training job is disabled or finished.";

    /**
     * use to stop job.
     */
    private static final Object STOP_OBJECT = new Object();
    private static final Logger LOGGER = Logger.getLogger(Common.class.toString());
    private static List<String> envTrustList = new ArrayList<>(Arrays.asList("x86", "android"));
    private static SecureRandom secureRandom;
    private static int iteration;
    private static boolean isHttps;

    /**
     * Generate the URL for device-sever interaction
     *
     * @param ifUseElb   whether a client randomly sends a request to a server address within a specified range.
     * @param serverNum  number of servers that can send requests.
     * @param domainName the URL for device-sever interaction set by user.
     * @return the URL for device-sever interaction.
     */
    public static String generateUrl(boolean ifUseElb, int serverNum, String domainName) {
        if (serverNum <= 0) {
            LOGGER.severe(Common.addTag("[generateUrl] the input argument <serverNum> is not valid: <= 0, it should " +
                    "be > 0, please check!"));
            throw new IllegalArgumentException();
        }
        String url;
        if ((domainName == null || domainName.isEmpty() || domainName.split("//").length < 2)) {
            LOGGER.severe(Common.addTag("[generateUrl] the input argument <domainName> is null or not valid, it " +
                    "should be like as https://...... or http://......  , please check!"));
            throw new IllegalArgumentException();
        }
        if (ifUseElb) {
            if (domainName.split("//")[1].split(":").length < 2) {
                LOGGER.severe(Common.addTag("[generateUrl] the format of <domainName> is not valid, it should be like" +
                        " as https://127.0.0.1:6666 or http://127.0.0.1:6666 when set useElb to true, please check!"));
                throw new IllegalArgumentException();
            }
            String ip = domainName.split("//")[1].split(":")[0];
            int port = Integer.parseInt(domainName.split("//")[1].split(":")[1]);
            if (!Common.checkIP(ip)) {
                LOGGER.severe(Common.addTag("[generateUrl] the <ip> split from domainName is not valid, domainName " +
                        "should be like as https://127.0.0.1:6666 or http://127.0.0.1:6666 when set useElb to true, " +
                        "please check!"));
                throw new IllegalArgumentException();
            }
            if (!Common.checkPort(port)) {
                LOGGER.severe(Common.addTag("[generateUrl] the <port> split from domainName is not valid, domainName " +
                        "should be like as https://127.0.0.1:6666 or http://127.0.0.1:6666 when set useElb to true, " +
                        "please check!"));
                throw new IllegalArgumentException();
            }
            String tag = domainName.split("//")[0] + "//";
            setIsHttps(domainName.split("//")[0].split(":")[0]);
            Random rand = new Random();
            int randomNum = rand.nextInt(100000) % serverNum + port;
            url = tag + ip + ":" + String.valueOf(randomNum);
        } else {
            url = domainName;
        }
        return url;
    }

    /**
     * Store weight name of classifier to a list.
     *
     * @param classifierWeightName the list to store weight name of classifier.
     */
    public static void setClassifierWeightName(List<String> classifierWeightName) {
        classifierWeightName.add("albert.pooler.weight");
        classifierWeightName.add("albert.pooler.bias");
        classifierWeightName.add("classifier.weight");
        classifierWeightName.add("classifier.bias");
        LOGGER.info(addTag("classifierWeightName size: " + classifierWeightName.size()));
    }

    /**
     * Store weight name of albert network to a list.
     *
     * @param albertWeightName the list to store weight name of albert network.
     */
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

    /**
     * Check whether the flName set by user is in the trust list.
     *
     * @param flName the model name set by user.
     * @return boolean value, true indicates the flName set by user is valid, false indicates the flName set by user
     * is not valid.
     */
    public static boolean checkFLName(String flName) {
        return (FL_NAME_TRUST_LIST.contains(flName));
    }

    /**
     * Check if the deploy environment set by user is in the trust list.
     *
     * @param env the deploy environment for federated learning task set by user.
     * @return boolean value, true indicates the deploy environment set by user is valid, false indicates the deploy
     * environment set by user is not valid.
     */
    public static boolean checkEnv(String env) {
        return (envTrustList.contains(env));
    }

    /**
     * Check whether the sslProtocol set by user is in the trust list.
     *
     * @param sslProtocol the ssl protocol set by user.
     * @return boolean value, true indicates the sslProtocol set by user is valid, false indicates the sslProtocol
     * set by user is not valid.
     */
    public static boolean checkSSLProtocol(String sslProtocol) {
        return (SSL_PROTOCOL_TRUST_LIST.contains(sslProtocol));
    }

    /**
     * The program waits for the specified time and then to continue.
     *
     * @param millis the waiting time (ms).
     */
    public static void sleep(long millis) {
        if (millis <= 0) {
            LOGGER.severe(addTag("[sleep] the millis is not valid(<= 0), will not do any thing, and stop the task"));
            throw new IllegalArgumentException();
        }
        if (millis > MAX_SLEEP_TIME) {
            LOGGER.severe(addTag("[sleep] the sleep time: " + millis + " exceed MAX_SLEEP_TIME: " + MAX_SLEEP_TIME
                    + "(unit: ms), will only sleep 30 minutes."));
            millis = MAX_SLEEP_TIME;
        }
        try {
            synchronized (STOP_OBJECT) {
                STOP_OBJECT.wait(millis);  // 1000 milliseconds is one second.
            }
        } catch (InterruptedException ex) {
            LOGGER.severe(addTag("[sleep] catch InterruptedException: " + ex.getMessage()));
        }
    }

    /**
     * Use to stop the wait method of a Object.
     */
    public static void notifyObject() {
        synchronized (STOP_OBJECT) {
            STOP_OBJECT.notify();
        }
    }

    /**
     * Get the waiting time for repeated requests.
     *
     * @param nextRequestTime the timestamp return from server.
     * @return the waiting time for repeated requests.
     */
    public static long getWaitTime(String nextRequestTime) {
        Date date = new Date();
        long currentTime = date.getTime();
        long waitTime = 0L;
        if (!(nextRequestTime == null || nextRequestTime.isEmpty())) {
            waitTime = Long.valueOf(nextRequestTime) - currentTime;
        }
        if (waitTime <= 0L) {
            LOGGER.severe(addTag("[getWaitTime] waitTime: " + waitTime + " is not valid (should be > 0), the reasons " +
                    "may be: the nextRequestTime <= currentTime, or the nextRequestTime is null, will stop the task"));
            throw new IllegalArgumentException();
        }
        LOGGER.info(addTag("[getWaitTime] next request time stamp: " + nextRequestTime + " current time stamp: " +
                currentTime));
        LOGGER.info(addTag("[getWaitTime] waitTime: " + waitTime));
        return waitTime;
    }

    /**
     * Get start time.
     *
     * @param tag the tag added to the logger.
     * @return start time.
     */
    public static long startTime(String tag) {
        Date startDate = new Date();
        long startTime = startDate.getTime();
        LOGGER.info(addTag("[start time] <" + tag + "> start time: " + startTime));
        return startTime;
    }

    /**
     * Get end time.
     *
     * @param start the start time.
     * @param tag   the tag added to the logger.
     */
    public static void endTime(long start, String tag) {
        Date endDate = new Date();
        long endTime = endDate.getTime();
        LOGGER.info(addTag("[end time] <" + tag + "> end time: " + endTime));
        LOGGER.info(addTag("[interval time] <" + tag + "> interval time(ms): " + (endTime - start)));
    }

    /**
     * Add specified tag to the message.
     *
     * @param message the message need to add tag.
     * @return the message after adding tag.
     */
    public static String addTag(String message) {
        return LOG_TITLE + message;
    }

    /**
     * Check whether the server is ready based on the message returned by the server.
     *
     * @param message the message returned by the server..
     * @return boolean value, true indicates the server is ready, false indicates the server is not ready.
     */
    public static boolean isSeverReady(byte[] message) {
        if (message == null) {
            LOGGER.severe(Common.addTag("[isSeverReady] the input argument <message> is null, please check!"));
            throw new IllegalArgumentException();
        }
        String messageStr = new String(message);
        if (messageStr.contains(SAFE_MOD)) {
            LOGGER.info(Common.addTag("[isSeverReady] " + SAFE_MOD + ", need wait some time and request again"));
            if (messageStr.split(":").length == 2) {
                iteration = Integer.parseInt(messageStr.split(":")[1]);
            } else {
                LOGGER.info(Common.addTag("[isSeverReady] the server does not return the current iteration."));
            }
            return false;
        } else if (messageStr.contains(JOB_NOT_AVAILABLE)) {
            LOGGER.info(Common.addTag("[isSeverReady] " + JOB_NOT_AVAILABLE + ", need wait some time and request " +
                    "again"));
            if (messageStr.split(":").length == 2) {
                iteration = Integer.parseInt(messageStr.split(":")[1]);
            } else {
                LOGGER.info(Common.addTag("[isSeverReady] the server does not return the current iteration."));
            }
            return false;
        } else {
            return true;
        }
    }

    /**
     * Convert a user-set path to a standard path.
     *
     * @param path the user-set path.
     * @return the standard path.
     */
    public static String getRealPath(String path) {
        if (path == null) {
            LOGGER.severe(Common.addTag("[getRealPath] the input argument <path> is null, please check!"));
            throw new IllegalArgumentException();
        }
        LOGGER.info(addTag("[getRealPath] original path: " + path));
        String[] paths = path.split(",");
        for (int i = 0; i < paths.length; i++) {
            if (paths[i] == null) {
                LOGGER.severe(Common.addTag("[getRealPath] the paths[i] is null, please check"));
                throw new IllegalArgumentException();
            }
            LOGGER.info(addTag("[getRealPath] original path " + i + ": " + paths[i]));
            File file = new File(paths[i]);
            try {
                paths[i] = file.getCanonicalPath();
            } catch (IOException e) {
                LOGGER.severe(addTag("[getRealPath] catch IOException in file.getCanonicalPath(): " + e.getMessage()));
                throw new IllegalArgumentException();
            }
        }
        String realPath = String.join(",", Arrays.asList(paths));
        LOGGER.info(addTag("[getRealPath] real path: " + realPath));
        return realPath;
    }

    /**
     * Check whether the path set by user exists.
     *
     * @param path the path set by user.
     * @return boolean value, true indicates the path is exist, false indicates the path does not exist
     */
    public static boolean checkPath(String path) {
        if (path == null) {
            LOGGER.severe(Common.addTag("[checkPath] the input argument <path> is null, please check!"));
            return false;
        }
        String[] paths = path.split(",");
        for (int i = 0; i < paths.length; i++) {
            if (paths[i] == null) {
                LOGGER.severe(Common.addTag("[checkPath] the paths[i] is null, please check"));
                return false;
            }
            LOGGER.info(addTag("[check path " + i + "] " + paths[i]));
            File file = new File(paths[i]);
            if (!file.exists()) {
                LOGGER.severe(Common.addTag("[checkPath] the path does not exist, please check"));
                return false;
            }
        }
        return true;
    }

    /**
     * Check whether the ip set by user is valid.
     *
     * @param ip the ip set by user.
     * @return boolean value, true indicates the ip is valid, false indicates the ip is not valid.
     */
    public static boolean checkIP(String ip) {
        if (ip == null) {
            LOGGER.severe(Common.addTag("[checkIP] the input argument <ip> is null, please check!"));
            throw new IllegalArgumentException();
        }
        String regex = "(25[0-4]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[1-9])[.]" +
                "(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])[.]" +
                "(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])[.]" +
                "(25[0-4]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[1-9])";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(ip);
        return matcher.matches();
    }

    /**
     * Check whether the port set by user is valid.
     *
     * @param port the port set by user.
     * @return boolean value, true indicates the port is valid, false indicates the port is not valid.
     */
    public static boolean checkPort(int port) {
        return port > 0 && port <= 65535;
    }

    /**
     * Obtain secure random.
     *
     * @return the secure random.
     */
    public static SecureRandom getSecureRandom() {
        if (secureRandom == null) {
            LOGGER.severe(Common.addTag("[setSecureRandom] the parameter secureRandom is null, please set it before " +
                    "use"));
            throw new IllegalArgumentException();
        }
        return secureRandom;
    }

    /**
     * Set the secure random to parameter secureRandom of the class Common.
     *
     * @param secureRandom the secure random.
     */
    public static void setSecureRandom(SecureRandom secureRandom) {
        if (secureRandom == null) {
            LOGGER.severe(Common.addTag("[setSecureRandom] the input parameter secureRandom is null, please check"));
            throw new IllegalArgumentException();
        }
        Common.secureRandom = secureRandom;
    }

    /**
     * Obtain fast secure random.
     *
     * @return the fast secure random.
     */
    public static SecureRandom getFastSecureRandom() {
        try {
            LOGGER.info(Common.addTag("[getFastSecureRandom] start create fastSecureRandom"));
            long start = System.currentTimeMillis();
            SecureRandom blockingRandom = SecureRandom.getInstanceStrong();
            boolean ifPredictionResistant = true;
            BlockCipher cipher = new AESEngine();
            int cipherLen = 256;
            int entropyBitsRequired = 384;
            byte[] nonce = null;
            boolean ifForceReseed = false;
            SecureRandom fastRandom = new SP800SecureRandomBuilder(blockingRandom, ifPredictionResistant)
                    .setEntropyBitsRequired(entropyBitsRequired)
                    .buildCTR(cipher, cipherLen, nonce, ifForceReseed);
            fastRandom.nextInt();
            LOGGER.info(Common.addTag("[getFastSecureRandom] finish create fastSecureRandom"));
            LOGGER.info(Common.addTag("[getFastSecureRandom] cost time: " + (System.currentTimeMillis() - start)));
            return fastRandom;
        } catch (NoSuchAlgorithmException e) {
            LOGGER.severe(Common.addTag("catch NoSuchAlgorithmException: " + e.getMessage()));
            throw new IllegalArgumentException();
        }
    }

    /**
     * Record the current iteration when server is not ready.
     *
     * @return int value, the current iteration when server is not ready.
     */
    public static int getIteration() {
        return iteration;
    }

    /**
     * Determine whether to conduct https communication according to the domain name set by user.
     *
     * @return boolean value, true means conducting https communication, false  means conducting http communication.
     */
    public static boolean isHttps() {
        return isHttps;
    }

    public static void setIsHttps(String tag) {
        if ("https".equals(tag)) {
            LOGGER.info(Common.addTag("conducting https communication"));
            Common.isHttps = true;
        } else if ("http".equals(tag)) {
            LOGGER.info(Common.addTag("conducting http communication"));
            Common.isHttps = false;
        } else {
            LOGGER.info(Common.addTag("The domain header set by the user is incorrect, please check"));
            throw new IllegalArgumentException();
        }
    }


    /**
     * Initialization session.
     *
     * @return the status code in client.
     */
    public static FLClientStatus initSession(String modelPath) {
        FLParameter flParameter = FLParameter.getInstance();
        LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
        if (Common.checkFLName(flParameter.getFlName())) {
            return deprecatedInitSession();
        }
        Status tag;
        LOGGER.info(Common.addTag("==========Loading model, " + modelPath + " Create " +
                " Session============="));
        Client client = ClientManager.getClient(flParameter.getFlName());
        tag = client.initSessionAndInputs(modelPath, localFLParameter.getMsConfig(), flParameter.getInputShape());
        if (!Status.SUCCESS.equals(tag)) {
            LOGGER.severe(Common.addTag("[initSession] unsolved error code in <initSessionAndInputs>: the return " +
                    "is -1"));
            return FLClientStatus.FAILED;
        }
        return FLClientStatus.SUCCESS;
    }

    /**
     * Free session.
     */
    protected static void freeSession() {
        FLParameter flParameter = FLParameter.getInstance();
        if (Common.checkFLName(flParameter.getFlName())) {
            deprecatedFreeSession();
        } else {
            LOGGER.info(Common.addTag("===========free session============="));
            Client client = ClientManager.getClient(flParameter.getFlName());
            client.free();
        }
    }

    /**
     * Initialization session.
     *
     * @return the status code in client.
     */
    private static FLClientStatus deprecatedInitSession() {
        FLParameter flParameter = FLParameter.getInstance();
        int tag = 0;
        if (flParameter.getFlName().equals(ALBERT)) {
            LOGGER.info(Common.addTag("==========Loading train model, " + flParameter.getTrainModelPath() + " Create " +
                    "Train Session============="));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            tag = alTrainBert.initSessionAndInputs(flParameter.getTrainModelPath(), true);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[initSession] unsolved error code in <initSessionAndInputs>: the return " +
                        "is -1"));
                return FLClientStatus.FAILED;
            }
            LOGGER.info(Common.addTag("==========Loading inference model, " + flParameter.getInferModelPath() + " " +
                    "Create inference Session============="));
            AlInferBert alInferBert = AlInferBert.getInstance();
            tag = alInferBert.initSessionAndInputs(flParameter.getInferModelPath(), false);
        } else if (flParameter.getFlName().equals(LENET)) {
            LOGGER.info(Common.addTag("==========Loading train model, " + flParameter.getTrainModelPath() + " Create " +
                    "Train Session============="));
            TrainLenet trainLenet = TrainLenet.getInstance();
            tag = trainLenet.initSessionAndInputs(flParameter.getTrainModelPath(), true);
        }
        if (tag == -1) {
            LOGGER.severe(Common.addTag("[initSession] unsolved error code in <initSessionAndInputs>: the return is " +
                    "-1"));
            return FLClientStatus.FAILED;
        }
        return FLClientStatus.SUCCESS;
    }

    /**
     * Free session.
     */
    private static void deprecatedFreeSession() {
        FLParameter flParameter = FLParameter.getInstance();
        if (flParameter.getFlName().equals(ALBERT)) {
            LOGGER.info(Common.addTag("===========free train session============="));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            SessionUtil.free(alTrainBert.getTrainSession());
            if (!flParameter.getTestDataset().equals("null")) {
                LOGGER.info(Common.addTag("===========free inference session============="));
                AlInferBert alInferBert = AlInferBert.getInstance();
                SessionUtil.free(alInferBert.getTrainSession());
            }
        } else if (flParameter.getFlName().equals(LENET)) {
            LOGGER.info(Common.addTag("===========free session============="));
            TrainLenet trainLenet = TrainLenet.getInstance();
            SessionUtil.free(trainLenet.getTrainSession());
        }
    }

    /**
     * Initialization session.
     *
     * @return the status code in client.
     */
    public static String getNextReqTime() {
        FLParameter flParameter = FLParameter.getInstance();
        long millis;
        if (flParameter.getSleepTime() != 0) {
            millis = flParameter.getSleepTime();
        } else {
            millis = SLEEP_TIME;
        }
        Date curDate = new Date();
        long currentTime = curDate.getTime();
        String nextRequestTime = Long.toString(currentTime + millis);
        return nextRequestTime;
    }
}
