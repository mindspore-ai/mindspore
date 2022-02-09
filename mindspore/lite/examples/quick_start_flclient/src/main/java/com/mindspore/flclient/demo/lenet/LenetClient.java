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

import com.mindspore.flclient.demo.common.ClassifierAccuracyCallback;
import com.mindspore.flclient.demo.common.PredictCallback;
import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.Client;
import com.mindspore.flclient.model.ClientManager;
import com.mindspore.flclient.model.DataSet;
import com.mindspore.flclient.model.LossCallback;
import com.mindspore.flclient.model.RunType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Defining the lenet client base class.
 *
 * @since v1.0
 */
public class LenetClient extends Client {
    private static final Logger LOGGER = Logger.getLogger(LenetClient.class.toString());
    private static final int NUM_OF_CLASS = 62;

    static {
        ClientManager.registerClient(new LenetClient());
    }

    @Override
    public List<Callback> initCallbacks(RunType runType, DataSet dataSet) {
        List<Callback> callbacks = new ArrayList<>();
        if (runType == RunType.TRAINMODE) {
            Callback lossCallback = new LossCallback(trainSession);
            callbacks.add(lossCallback);
        } else if (runType == RunType.EVALMODE) {
            if (dataSet instanceof LenetDataSet) {
                Callback evalCallback = new ClassifierAccuracyCallback(trainSession, dataSet.batchSize, NUM_OF_CLASS,
                        ((LenetDataSet) dataSet).getTargetLabels());
                callbacks.add(evalCallback);
            }
        } else {
            Callback inferCallback = new PredictCallback(trainSession, dataSet.batchSize, NUM_OF_CLASS);
            callbacks.add(inferCallback);
        }
        return callbacks;
    }

    @Override
    public Map<RunType, Integer> initDataSets(Map<RunType, List<String>> files) {
        Map<RunType, Integer> sampleCounts = new HashMap<>();
        List<String> trainFiles = files.getOrDefault(RunType.TRAINMODE, null);
        if (trainFiles != null) {
            DataSet trainDataSet = new LenetDataSet(NUM_OF_CLASS);
            trainDataSet.init(trainFiles);
            dataSets.put(RunType.TRAINMODE, trainDataSet);
            sampleCounts.put(RunType.TRAINMODE, trainDataSet.sampleSize);
        }
        List<String> evalFiles = files.getOrDefault(RunType.EVALMODE, null);
        if (evalFiles != null) {
            LenetDataSet evalDataSet = new LenetDataSet(NUM_OF_CLASS);
            evalDataSet.init(evalFiles);
            dataSets.put(RunType.EVALMODE, evalDataSet);
            sampleCounts.put(RunType.EVALMODE, evalDataSet.sampleSize);
        }
        List<String> inferFiles = files.getOrDefault(RunType.INFERMODE, null);
        if (inferFiles != null) {
            DataSet inferDataSet = new LenetDataSet(NUM_OF_CLASS);
            inferDataSet.init(inferFiles);
            dataSets.put(RunType.INFERMODE, inferDataSet);
            sampleCounts.put(RunType.INFERMODE, inferDataSet.sampleSize);
        }
        return sampleCounts;
    }

    @Override
    public float getEvalAccuracy(List<Callback> evalCallbacks) {
        for (Callback callBack : evalCallbacks) {
            if (callBack instanceof ClassifierAccuracyCallback) {
                return ((ClassifierAccuracyCallback) callBack).getAccuracy();
            }
        }
        LOGGER.severe("don not find accuracy related callback");
        return Float.NaN;
    }

    @Override
    public List<Object> getInferResult(List<Callback> inferCallbacks) {
        DataSet inferDataSet = dataSets.getOrDefault(RunType.INFERMODE, null);
        if (inferDataSet == null) {
            return new ArrayList<>();
        }
        for (Callback callBack : inferCallbacks) {
            if (callBack instanceof PredictCallback) {
                return ((PredictCallback) callBack).getPredictResults().subList(0, inferDataSet.sampleSize);
            }
        }
        LOGGER.severe("don not find accuracy related callback");
        return new ArrayList<>();
    }
}
