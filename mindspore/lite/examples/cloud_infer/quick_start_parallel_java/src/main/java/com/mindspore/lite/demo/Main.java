/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

package com.mindspore.lite.demo;
import com.mindspore.config.DataType;
import com.mindspore.config.DeviceType;
import com.mindspore.config.MSContext;
import com.mindspore.config.RunnerConfig;
import com.mindspore.ModelParallelRunner;
import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.config.ModelType;
import com.mindspore.config.Version;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Random;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    private static ModelParallelRunner runner;
    private static List<MSTensor> inputs;
    private static List<MSTensor> outputs;

    public static float[] generateArray(int len) {
        Random rand = new Random();
        float[] arr = new float[len];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = rand.nextFloat();
        }
        return arr;
    }

    private static ByteBuffer floatArrayToByteBuffer(float[] floats) {
        if (floats == null) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocateDirect(floats.length * Float.BYTES);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.put(floats);
        return buffer;
    }

    private static void freeTensor(){
        for (int i = 0; i < inputs.size(); i++) {
            inputs.get(i).free();
        }
        for (int i = 0; i < outputs.size(); i++) {
            outputs.get(i).free();
        }
    }

    public static void main(String[] args) {
        System.out.println(Version.version());
        if (args.length < 1) {
            System.err.println("The model path parameter must be passed.");
            return;
        }
        String modelPath = args[0];

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

        // init input tensor
        inputs = runner.getInputs();
        if (inputs.size() != 1) {
            System.err.println("inputs size is wrong.");
            return;
        }
        MSTensor input = inputs.get(0);
        if (input.getDataType() != DataType.kNumberTypeFloat32) {
            System.err.println("Input tensor data type is not float, the data type is " + input.getDataType());
            return;
        }
        // Generator Random Data.
        int elementNums = input.elementsNum();
        float[] randomData = generateArray(elementNums);
        ByteBuffer inputData = floatArrayToByteBuffer(randomData);
        input.setData(inputData);

        // init output
        outputs = runner.getOutputs();
        if (outputs.size() != 1) {
            System.err.println("outputs size is wrong.");
            return;
        }
        List<MSTensor> outputs = new ArrayList<>();

        // runner do predict
        ret = runner.predict(inputs,outputs);
        if (!ret) {
            System.err.println("MindSpore Lite predict failed.");
            freeTensor();
            runner.free();
            return;
        }
        System.out.println("========== model parallel runner predict success ==========");
        config.free();
        freeTensor();
        runner.free();
    }
}
