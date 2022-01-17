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

package com.mindspore.lite.train_lenet;

import com.mindspore.Graph;
import com.mindspore.Model;
import com.mindspore.config.DeviceType;
import com.mindspore.config.MSContext;
import com.mindspore.config.TrainCfg;
import com.mindspore.MSTensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class NetRunner {
    private Model model;
    private int batchSize;
    private long dataSize; // one input data size, in byte
    private final DataSet ds = new DataSet();
    private long numOfClasses;
    private final long cycles = 2000;
    private int idx = 1;
    private int virtualBatch = 16;
    private ByteBuffer imageInputBuf;
    private ByteBuffer labelInputBuf;
    private int imageBatchElements;
    private MSTensor imageTensor;
    private MSTensor labelTensor;
    private int[] targetLabels;

    private int initInputs() {
        List<MSTensor> inputs = model.getInputs();
        if (inputs.size() <= 1) {
            System.err.println("model input size: " + inputs.size());
            return -1;
        }

        int dataIndex = 0;
        int labelIndex = 1;
        batchSize = inputs.get(dataIndex).getShape()[0];
        dataSize = inputs.get(dataIndex).size() / batchSize;
        System.out.println("batch_size: " + batchSize);
        System.out.println("virtual batch multiplier: " + virtualBatch);

        imageTensor = inputs.get(dataIndex);
        imageInputBuf = ByteBuffer.allocateDirect((int) imageTensor.size());
        imageInputBuf.order(ByteOrder.nativeOrder());
        imageBatchElements = inputs.get(dataIndex).elementsNum();

        labelTensor = inputs.get(labelIndex);
        labelInputBuf = ByteBuffer.allocateDirect((int) labelTensor.size());
        labelInputBuf.order(ByteOrder.nativeOrder());
        targetLabels = new int[batchSize];
        return 0;
    }

    public int initAndFigureInputs(String modelPath, int virtualBatchSize) {
        System.out.println("Model path is " + modelPath);
        MSContext context = new MSContext();
        // use default param init context
        context.init();
        boolean isSuccess = context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        if (!isSuccess) {
            System.err.println("Load graph failed");
            context.free();
            return -1;
        }
        TrainCfg trainCfg = new TrainCfg();
        isSuccess = trainCfg.init();
        if (!isSuccess) {
            System.err.println("Init train config failed");
            context.free();
            trainCfg.free();
            return -1;
        }
        model = new Model();
        Graph graph = new Graph();
        isSuccess = graph.load(modelPath);
        if (!isSuccess) {
            System.err.println("Load graph failed");
            graph.free();
            context.free();
            trainCfg.free();
            return -1;
        }
        isSuccess = model.build(graph, context, trainCfg);
        if (!isSuccess) {
            System.err.println("Build model failed");
            return -1;
        }
        virtualBatch = virtualBatchSize;
        model.setupVirtualBatch(virtualBatch, 0.01f, 1.00f);
        return initInputs();
    }

    public int initDB(String datasetPath) {
        if (dataSize != 0) {
            ds.setExpectedDataSize(dataSize);
        }
        ds.initializeMNISTDatabase(datasetPath);
        numOfClasses = ds.getNumOfClasses();
        if (numOfClasses != 10) {
            System.err.println("unexpected num_of_class: " + numOfClasses);
            System.exit(1);
        }

        if (ds.testData.size() == 0) {
            System.err.println("test data size is 0");
            return -1;
        }

        return 0;
    }

    public float getLoss() {
        MSTensor tensor = searchOutputsForSize(1);
        if (tensor == null) {
            System.err.println("get loss tensor failed");
            return Float.NaN;
        }
        return tensor.getFloatData()[0];
    }

    private MSTensor searchOutputsForSize(int size) {
        List<MSTensor> outputs = model.getOutputs();
        for (MSTensor tensor : outputs) {
            if (tensor.elementsNum() == size) {
                return tensor;
            }
        }
        System.err.println("can not find output the tensor which element num is " + size);
        return null;
    }

    public int trainLoop() {
        boolean isSuccess = model.setTrainMode(true);
        if (!isSuccess) {
            model.free();
            System.err.println("set train mode failed");
            return -1;
        }
        float min_loss = 1000;
        float max_acc = 0;
        for (int i = 0; i < cycles; i++) {
            for (int b = 0; b < virtualBatch; b++) {
                fillInputData(ds.getTrainData(), false);
                isSuccess = model.runStep();
                if (!isSuccess) {
                    model.free();
                    System.err.println("run step failed");
                    return -1;
                }
                float loss = getLoss();
                if (min_loss > loss) {
                    min_loss = loss;
                }
                if ((b == 0) && ((i + 1) % 500 == 0)) {
                    float acc = calculateAccuracy(10); // only test 10 batch size
                    if (max_acc < acc) {
                        max_acc = acc;
                    }
                    System.out.println("step_" + (i + 1) + ": \tLoss is " + loss + " [min=" + min_loss + "]" + " " +
                            "max_acc=" + max_acc);
                }
            }
        }
        return 0;
    }

    public float calculateAccuracy(long maxTests) {
        float accuracy = 0;
        Vector<DataSet.DataLabelTuple> test_set = ds.getTestData();
        long tests = test_set.size() / batchSize;
        if (maxTests != -1 && tests < maxTests) {
            tests = maxTests;
        }
        model.setTrainMode(false);
        for (long i = 0; i < tests; i++) {
            int[] labels = fillInputData(test_set, (maxTests == -1));
            model.predict();
            MSTensor outputsv = searchOutputsForSize((int) (batchSize * numOfClasses));
            if (outputsv == null) {
                System.err.println("can not find output tensor with size: " + batchSize * numOfClasses);
                model.free();
                System.exit(1);
            }
            float[] scores = outputsv.getFloatData();
            for (int b = 0; b < batchSize; b++) {
                int max_idx = 0;
                float max_score = scores[(int) (numOfClasses * b)];
                for (int c = 0; c < numOfClasses; c++) {
                    if (scores[(int) (numOfClasses * b + c)] > max_score) {
                        max_score = scores[(int) (numOfClasses * b + c)];
                        max_idx = c;
                    }

                }
                if (labels[b] == max_idx) {
                    accuracy += 1.0;
                }
            }
        }
        model.setTrainMode(true);
        accuracy /= (batchSize * tests);
        return accuracy;
    }

    // each time fill batch_size data
    int[] fillInputData(Vector<DataSet.DataLabelTuple> dataset, boolean serially) {
        int totalSize = dataset.size();
        imageInputBuf.clear();
        labelInputBuf.clear();
        for (int i = 0; i < batchSize; i++) {
            if (serially) {
                idx = (++idx) % totalSize;
            } else {
                idx = (int) (Math.random() * totalSize);
            }
            DataSet.DataLabelTuple dataLabelTuple = dataset.get(idx);
            byte[] inputBatchData = dataLabelTuple.data;
            for (int j = 0; j < imageBatchElements / batchSize; j++) {
                imageInputBuf.putFloat((inputBatchData[j] & 0xff) / 255.0f);
            }
            int label = dataLabelTuple.label;
            labelInputBuf.putInt(label & 0xff);
            targetLabels[i] = label;
        }
        imageTensor.setData(imageInputBuf);
        labelTensor.setData(labelInputBuf);
        return targetLabels;
    }

    public void trainModel(String modelPath, String datasetPath, int virtualBatch) {
        int index = modelPath.lastIndexOf(".ms");
        if (index == -1) {
            System.err.println("The model " + modelPath + " should be named *.ms");
            return;
        }
        System.out.println("==========Loading Model, Create Train Session=============");
        int ret = initAndFigureInputs(modelPath, virtualBatch);
        if (ret != 0) {
            System.out.println("==========Init and figure inputs failed================");
            model.free();
            return;
        }
        System.out.println("==========Initing DataSet================");
        ret = initDB(datasetPath);
        if (ret != 0) {
            System.out.println("==========Init dataset failed================");
            return;
        }
        System.out.println("==========Training Model===================");
        ret = trainLoop();
        if (ret != 0) {
            System.out.println("==========Init dataset failed================");
            model.free();
            return;
        }
        System.out.println("==========Evaluating The Trained Model============");
        float acc = calculateAccuracy(-1);
        System.out.println("accuracy = " + acc);

        if (cycles > 0) {
            // arg 0: FileName
            // arg 1: quantization type QT_DEFAULT -> 0
            // arg 2: model type MT_TRAIN -> 0
            // arg 3: use default output tensor names
            String trainedFilePath = modelPath.substring(0, index) + "_trained.ms";
            if (model.export(trainedFilePath, 0, false, new ArrayList<>())) {
                System.out.println("Trained model successfully saved: " + trainedFilePath);
            } else {
                System.err.println("Save model error.");
            }
        }
        model.free();
    }

}
