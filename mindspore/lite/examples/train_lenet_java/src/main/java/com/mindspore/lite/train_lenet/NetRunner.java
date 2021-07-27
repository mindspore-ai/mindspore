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

import com.mindspore.lite.MSTensor;
import com.mindspore.lite.LiteSession;
import com.mindspore.lite.TrainSession;
import com.mindspore.lite.config.MSConfig;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Map;
import java.util.Vector;

public class NetRunner {
    private int dataIndex = 0;
    private int labelIndex = 1;
    private LiteSession session;
    private long batchSize;
    private long dataSize; // one input data size, in byte
    private DataSet ds = new DataSet();
    private long numOfClasses;
    private long cycles = 2000;
    private int idx = 1;
    private int virtualBatch = 16;
    private String trainedFilePath = "trained.ms";

    public void initAndFigureInputs(String modelPath, int virtualBatchSize) {
        MSConfig msConfig = new MSConfig();
        // arg 0: DeviceType:DT_CPU -> 0
        // arg 1: ThreadNum -> 2
        // arg 2: cpuBindMode:NO_BIND ->  0
        // arg 3: enable_fp16 -> false
        msConfig.init(0, 2, 0, false);
        session = new LiteSession();
        System.out.println("Model path is " + modelPath);
        session = TrainSession.createTrainSession(modelPath, msConfig, false);
        virtualBatch = virtualBatchSize;
        session.setupVirtualBatch(virtualBatch, 0.01f, 1.00f);

        List<MSTensor> inputs = session.getInputs();
        if (inputs.size() <= 1) {
            System.err.println("model input size: " + inputs.size());
            return;
        }

        dataIndex = 0;
        labelIndex = 1;
        batchSize = inputs.get(dataIndex).getShape()[0];
        dataSize = inputs.get(dataIndex).size() / batchSize;
        System.out.println("batch_size: " + batchSize);
        System.out.println("virtual batch multiplier: " + virtualBatch);
        int index = modelPath.lastIndexOf(".ms");
        if (index == -1) {
            System.out.println("The model " + modelPath + " should be named *.ms");
            return;
        }
        trainedFilePath = modelPath.substring(0, index) + "_trained.ms";
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
        return tensor.getFloatData()[0];
    }

    private MSTensor searchOutputsForSize(int size) {
        Map<String, MSTensor> outputs = session.getOutputMapByTensor();
        for (MSTensor tensor : outputs.values()) {
            if (tensor.elementsNum() == size) {
                return tensor;
            }
        }
        System.err.println("can not find output the tensor which element num is " + size);
        return null;
    }

    public int trainLoop() {
        session.train();
        float min_loss = 1000;
        float max_acc = 0;
        for (int i = 0; i < cycles; i++) {
            for (int b = 0; b < virtualBatch; b++) {
                fillInputData(ds.getTrainData(), false);
                session.runGraph();
                float loss = getLoss();
                if (min_loss > loss) {
                    min_loss = loss;
                }
                if ((b == 0) && ((i + 1) % 500 == 0)) {
                    float acc = calculateAccuracy(10); // only test 10 batch size
                    if (max_acc < acc) {
                        max_acc = acc;
                    }
                    System.out.println("step_" + (i + 1) + ": \tLoss is " + loss + " [min=" + min_loss + "]" + " max_accc=" + max_acc);
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
        session.eval();
        for (long i = 0; i < tests; i++) {
            Vector<Integer> labels = fillInputData(test_set, (maxTests == -1));
            if (labels.size() != batchSize) {
                System.err.println("unexpected labels size: " + labels.size() + " batch_size size: " + batchSize);
                System.exit(1);
            }
            session.runGraph();
            MSTensor outputsv = searchOutputsForSize((int) (batchSize * numOfClasses));
            if (outputsv == null) {
                System.err.println("can not find output tensor with size: " + batchSize * numOfClasses);
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
                if (labels.get(b) == max_idx) {
                    accuracy += 1.0;
                }
            }
        }
        session.train();
        accuracy /= (batchSize * tests);
        return accuracy;
    }

    // each time fill batch_size data
    Vector<Integer> fillInputData(Vector<DataSet.DataLabelTuple> dataset, boolean serially) {
        Vector<Integer> labelsVec = new Vector<Integer>();
        int totalSize = dataset.size();

        List<MSTensor> inputs = session.getInputs();

        int inputDataCnt = inputs.get(dataIndex).elementsNum();
        float[] inputBatchData = new float[inputDataCnt];

        int labelDataCnt = inputs.get(labelIndex).elementsNum();
        int[] labelBatchData = new int[labelDataCnt];

        for (int i = 0; i < batchSize; i++) {
            if (serially) {
                idx = (++idx) % totalSize;
            } else {
                idx = (int) (Math.random() * totalSize);
            }

            int label = 0;
            DataSet.DataLabelTuple dataLabelTuple = dataset.get(idx);
            label = dataLabelTuple.label;
            System.arraycopy(dataLabelTuple.data, 0, inputBatchData, (int) (i * dataLabelTuple.data.length), dataLabelTuple.data.length);
            labelBatchData[i] = label;
            labelsVec.add(label);
        }

        ByteBuffer byteBuf = ByteBuffer.allocateDirect(inputBatchData.length * Float.BYTES);
        byteBuf.order(ByteOrder.nativeOrder());
        for (int i = 0; i < inputBatchData.length; i++) {
            byteBuf.putFloat(inputBatchData[i]);
        }
        inputs.get(dataIndex).setData(byteBuf);

        ByteBuffer labelByteBuf = ByteBuffer.allocateDirect(labelBatchData.length * 4);
        labelByteBuf.order(ByteOrder.nativeOrder());
        for (int i = 0; i < labelBatchData.length; i++) {
            labelByteBuf.putInt(labelBatchData[i]);
        }
        inputs.get(labelIndex).setData(labelByteBuf);

        return labelsVec;
    }

    public void trainModel(String modelPath, String datasetPath, int virtualBatch) {
        System.out.println("==========Loading Model, Create Train Session=============");
        initAndFigureInputs(modelPath, virtualBatch);
        System.out.println("==========Initing DataSet================");
        initDB(datasetPath);
        System.out.println("==========Training Model===================");
        trainLoop();
        System.out.println("==========Evaluating The Trained Model============");
        float acc = calculateAccuracy(-1);
        System.out.println("accuracy = " + acc);

        if (cycles > 0) {
            // arg 0: FileName
            // arg 1: model type MT_TRAIN -> 0
            // arg 2: quantization type QT_DEFAULT -> 0
            if (session.export(trainedFilePath, 0, 0)) {
                System.out.println("Trained model successfully saved: " + trainedFilePath);
            } else {
                System.err.println("Save model error.");
            }
        }
        session.free();
    }

}
