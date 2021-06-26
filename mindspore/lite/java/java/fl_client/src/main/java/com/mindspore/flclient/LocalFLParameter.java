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

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

public class LocalFLParameter {
    private static final Logger LOGGER = Logger.getLogger(LocalFLParameter.class.toString());
    public static final int SEED_SIZE = 32;
    public static final int IVEC_LEN = 16;
    public static final String LENET = "lenet";
    public static final String ADBERT = "adbert";
    private List<String> classifierWeightName = new ArrayList<>();
    private List<String> albertWeightName = new ArrayList<>();

    private String flID;
    private String encryptLevel = "NotEncrypt";
    private String earlyStopMod = "NotEarlyStop";
    private String serverMod = ServerMod.HYBRID_TRAINING.toString();
    private String safeMod = "The cluster is in safemode.";

    private static volatile LocalFLParameter localFLParameter;

    private LocalFLParameter() {
        // set classifierWeightName albertWeightName
        Common.setClassifierWeightName(classifierWeightName);
        Common.setAlbertWeightName(albertWeightName);
    }

    public static synchronized LocalFLParameter getInstance() {
        LocalFLParameter localRef = localFLParameter;
        if (localRef == null) {
            synchronized (LocalFLParameter.class) {
                localRef = localFLParameter;
                if (localRef == null) {
                    localFLParameter = localRef = new LocalFLParameter();
                }
            }
        }
        return localRef;
    }

    public List<String> getClassifierWeightName() {
        if (classifierWeightName.isEmpty()) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <classifierWeightName> is null, please set it before use"));
            throw new RuntimeException();
        }
        return classifierWeightName;
    }

    public void setClassifierWeightName(List<String> classifierWeightName) {
        this.classifierWeightName = classifierWeightName;
    }

    public List<String> getAlbertWeightName() {
        if (albertWeightName.isEmpty()) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <classifierWeightName> is null, please set it before use"));
            throw new RuntimeException();
        }
        return albertWeightName;
    }

    public void setAlbertWeightName(List<String> albertWeightName) {
        this.albertWeightName = albertWeightName;
    }

    public String getFlID() {
        if ("".equals(flID) || flID == null) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <flID> is null, please set it before use"));
            throw new RuntimeException();
        }
        return flID;
    }

    public void setFlID(String flID) {
        this.flID = flID;
    }

    public EncryptLevel getEncryptLevel() {
        return EncryptLevel.valueOf(encryptLevel);
    }

    public void setEncryptLevel(String encryptLevel) {
        this.encryptLevel = encryptLevel;
    }

    public EarlyStopMod getEarlyStopMod() {
        return EarlyStopMod.valueOf(earlyStopMod);
    }

    public void setEarlyStopMod(String earlyStopMod) {
        this.earlyStopMod = earlyStopMod;
    }

    public String getServerMod() {
        return serverMod;
    }

    public void setServerMod(String serverMod) {
        this.serverMod = serverMod;
    }

    public String getSafeMod() {
        return safeMod;
    }

    public void setSafeMod(String safeMod) {
        this.safeMod = safeMod;
    }
}
