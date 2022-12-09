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

import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.config.DataType;
import com.mindspore.config.DeviceType;
import com.mindspore.config.MSContext;
import com.mindspore.config.ModelType;
import com.mindspore.config.Version;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Random;

public class Main {
    private static Model model;

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

    private static boolean compile(String modelPath) {
        MSContext context = new MSContext();
        // use default param init context
        context.init();
        boolean ret = context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        if (!ret) {
            System.err.println("Compile graph failed");
            context.free();
            return false;
        }
        // Create the MindSpore lite session.
        model = new Model();
        // Compile graph.
        ret = model.build(modelPath, ModelType.MT_MINDIR, context);
        if (!ret) {
            System.err.println("Compile graph failed");
            model.free();
            return false;
        }
        return true;
    }

    private static boolean run() {
        // Get input tesnor using getInputs API.
        MSTensor inputTensor = model.getInputs().get(0);
        if (inputTensor.getDataType() != DataType.kNumberTypeFloat32) {
            System.err.println("Input tensor data type is not float, the data type is " + inputTensor.getDataType());
            return false;
        }
        // Generator Random Data.
        int elementNums = inputTensor.elementsNum();
        float[] randomData = generateArray(elementNums);
        ByteBuffer inputData = floatArrayToByteBuffer(randomData);

        // Set Input Data.
        inputTensor.setData(inputData);

        // Run Inference.
        boolean ret = model.predict();
        if (!ret) {
            inputTensor.free();
            System.err.println("MindSpore Lite run failed.");
            return false;
        }

        // Get Output Tensor Data.
        MSTensor outTensor = model.getOutputs().get(0);

        // Print out Tensor Data.
        StringBuilder msgSb = new StringBuilder();
        msgSb.append("out tensor shape: [");
        int[] shape = outTensor.getShape();
        for (int dim : shape) {
            msgSb.append(dim).append(",");
        }
        msgSb.append("]");
        if (outTensor.getDataType() != DataType.kNumberTypeFloat32) {
            inputTensor.free();
            outTensor.free();
            System.err.println("output tensor data type is not float, the data type is " + outTensor.getDataType());
            return false;
        }
        float[] result = outTensor.getFloatData();
        if (result == null) {
            inputTensor.free();
            outTensor.free();
            System.err.println("decodeBytes return null");
            return false;
        }
        msgSb.append(" and out data:");
        for (int i = 0; i < 50 && i < outTensor.elementsNum(); i++) {
            msgSb.append(" ").append(result[i]);
        }
        System.out.println(msgSb.toString());
        // In/Out Tensor must free
        inputTensor.free();
        outTensor.free();
        return true;
    }

    private static void freeBuffer() {
        model.free();
    }

    public static void main(String[] args) {
        System.out.println(Version.version());
        if (args.length < 1) {
            System.err.println("The model path parameter must be passed.");
            return;
        }
        String modelPath = args[0];
        boolean ret = compile(modelPath);
        if (!ret) {
            System.err.println("MindSpore Lite compile failed.");
            return;
        }

        ret = run();
        if (!ret) {
            System.err.println("MindSpore Lite run failed.");
            freeBuffer();
            return;
        }

        freeBuffer();
    }
}
