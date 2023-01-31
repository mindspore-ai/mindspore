/*
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

import com.mindspore.MSTensor;
import com.mindspore.config.DeviceType;
import com.mindspore.config.MSContext;
import com.mindspore.config.DataType;
import com.mindspore.config.RunnerConfig;
import com.mindspore.Model;
import com.mindspore.ModelParallelRunner;
import java.util.List;
import java.util.ArrayList;

import java.io.*;

public class Benchmark {
    private static Model model;

    public static byte[] readBinFile(String fileName, int size) {
        try {
            DataInputStream is = new DataInputStream(
                    new BufferedInputStream(new FileInputStream(
                            fileName)));
            byte[] buf = new byte[size];
            is.read(buf);
            is.close();
            return buf;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static boolean compareData(String filePath, float accuracy) {
        double meanError = 0;
        File file = new File(filePath);
        if (file.exists()) {
            try {
                FileReader fileReader = new FileReader(file);
                BufferedReader br = new BufferedReader(fileReader);
                String lineContent = null;

                int line = 0;
                MSTensor outTensor = null;
                String name = null;
                while ((lineContent = br.readLine()) != null) {
                    String[] strings = lineContent.split(" ");
                    if (line++ % 2 == 0) {
                        name = strings[0];
                        outTensor = model.getOutputByTensorName(name);
                        continue;
                    }
                    float[] benchmarkData = new float[strings.length];
                    for (int i = 0; i < strings.length; i++) {
                        benchmarkData[i] = Float.parseFloat(strings[i]);
                    }
                    float[] outData = outTensor.getFloatData();
                    int errorCount = 0;
                    for (int i = 0; i < benchmarkData.length; i++) {
                        double relativeTolerance = 1e-5;
                        double absoluteTolerance = 1e-8;
                        double tolerance = absoluteTolerance + relativeTolerance * Math.abs(benchmarkData[i]);
                        double absoluteError = Math.abs(outData[i] - benchmarkData[i]);
                        if (absoluteError > tolerance) {
                            if (Math.abs(benchmarkData[i] - 0.0f) < Float.MIN_VALUE)
                                if (absoluteError > 1e-5) {
                                    meanError += absoluteError;
                                    errorCount++;
                                } else {
                                    continue;
                                }
                        } else {
                            meanError += absoluteError / (Math.abs(benchmarkData[i]) + Float.MIN_VALUE);
                            errorCount++;
                        }
                    }

                    if (meanError > 0.0f) {
                        meanError /= errorCount;
                    }
                    if (meanError <= 0.0000001) {
                        System.out.println("Mean bias of node/tensor " + name + " : 0%");
                    } else {
                        System.out.println("Mean bias of node/tensor " + name + " : " + meanError * 100 + "%");
                    }
                }
                br.close();
                fileReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return meanError < accuracy;
    }
    
    public static void main(String[] args) {
        if (args.length < 4) {
            System.err.println("We must pass parameters such as modelPath, inDataFile, benchmarkDataFile and accuracy.");
            return;
        }

        String modelPath = args[0];
        String[] inDataFile = args[1].split(",");
        String benchmarkDataFile = args[2];
        float accuracy = Float.parseFloat(args[3]);
        if (args.length == 5 && args[4].equals("Runner")) {
            // use default param init context
            MSContext context = new MSContext();
            context.init(1,0);
            boolean ret = context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
            if (!ret) {
                System.err.println("init context failed");
                context.free();
                return ;
            }
            // init runner config
            RunnerConfig config = new RunnerConfig();
            config.init(context);
            config.setWorkersNum(2);
            // init ModelParallelRunner
            ModelParallelRunner runner = new ModelParallelRunner();
            ret = runner.init(modelPath, config);
            if (!ret) {
                System.err.println("ModelParallelRunner init failed.");
                runner.free();
                return;
            }
            List<MSTensor> inputs = runner.getInputs();
            for (int index = 0; index < inputs.size(); index++) {
                MSTensor msTensor = inputs.get(index);
                if (msTensor.getDataType() != DataType.kNumberTypeFloat32) {
                    System.err.println("Input tensor data type is not float, the data type is " + msTensor.getDataType());
                    runner.free();
                    return;
                }
                // Set Input Data.
                byte[] data = readBinFile(inDataFile[index], (int) msTensor.size());
                msTensor.setData(data);
            }
            // init output
            List<MSTensor> outputs = new ArrayList<>();

            // runner do predict
            ret = runner.predict(inputs,outputs);
            if (!ret) {
                System.err.println("MindSpore Lite predict failed.");
                runner.free();
                return;
            }
            System.out.println("========== model parallel runner predict success ==========");
            config.free();
            for (int i = 0; i < inputs.size(); i++) {
                inputs.get(i).free();
            }
            for (int i = 0; i < outputs.size(); i++) {
                outputs.get(i).free();
            }
            runner.free();
            return;
        }

        MSContext context = new MSContext();
        context.init(1, 0);
        boolean ret = context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        if (!ret) {
            System.err.println("Compile graph failed");
            return;
        }
        model = new Model();
        ret = model.build(modelPath, 0, context);
        if (!ret) {
            System.err.println("Compile graph failed, model path is " + modelPath);
            model.free();
            return;
        }
        for (int index = 0; index < model.getInputs().size(); index++) {
            MSTensor msTensor = model.getInputs().get(index);
            if (msTensor.getDataType() != DataType.kNumberTypeFloat32) {
                System.err.println("Input tensor data type is not float, the data type is " + msTensor.getDataType());
                model.free();
                return;
            }
            // Set Input Data.
            byte[] data = readBinFile(inDataFile[index], (int) msTensor.size());
            msTensor.setData(data);
        }

        // Run Inference.
        ret = model.predict();
        if (!ret) {
            System.err.println("MindSpore Lite run failed.");
            model.free();
            return;
        }

        boolean benchmarkResult = compareData(benchmarkDataFile, accuracy);
        model.free();
        if (!benchmarkResult) {
            System.err.println(modelPath + " accuracy error is too large.");
            System.exit(1);
        }
    }
}
