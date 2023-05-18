/*
 * Copyright 2023 Huawei Technologies Co., Ltd
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
import com.mindspore.config.AscendDeviceInfo;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.sql.DataTruncation;
import java.util.*;

public class Main {
    private static List<Model> models;

    public static int[] generateIntArray(int len) {
        Random rand = new Random();
        int[] arr = new int[len];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = rand.nextInt(10);
        }
        return arr;
    }

    public static byte[] generateByteArray(int len) {
        Random rand = new Random();
        byte[] arr = new byte[len];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (byte)0;
        }
        return arr;
    }

    private static ByteBuffer intArrayToByteBuffer(int[] ints) {
        if (ints == null) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocateDirect(ints.length * Integer.BYTES);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (int i : ints) {
            buffer.putInt(i);
        }
        return buffer;
    }

    private static ByteBuffer byteArrayToByteBuffer(byte[] bytes) {
        if (bytes == null) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocateDirect(bytes.length * Byte.BYTES);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(bytes);
        return buffer;
    }

    private static boolean compile(String modelPath, String incModelPath, int deviceId, int rankId, String config) {
        AscendDeviceInfo ascendDeviceInfo = new AscendDeviceInfo();
        ascendDeviceInfo.setProvider("ge");
        ascendDeviceInfo.setDeviceID(deviceId);
        ascendDeviceInfo.setRankID(rankId);

        MSContext context = new MSContext();
        // use default param init context
        context.init();
        boolean ret = context.addDeviceInfo(ascendDeviceInfo);
        if (!ret) {
            System.err.println("Compile add device info failed");
            context.free();
            return false;
        }
        // Build main and incremental models.
        models = Model.build(modelPath, incModelPath, ModelType.MT_MINDIR, context, config);
        if (models.size() == 0) {
            System.err.println("Compile graph failed");
            context.free();
            return false;
        }

        context.free();
        return true;
    }

    private static boolean run(boolean isInc) {
        int mdlIndex = isInc ? 1 : 0;

        // Get input tesnor using getInputs API.
        List<MSTensor> tensors = models.get(mdlIndex).getInputs();
        for (int i = 0; i < tensors.size(); i++) {
            MSTensor tensor = tensors.get(i);
            int elementNums = tensor.elementsNum();
            int dType = tensor.getDataType();
            switch (dType) {
                case DataType.kNumberTypeInt:
                case DataType.kNumberTypeInt32: {
                    int[] randIntData = generateIntArray(elementNums);
                    ByteBuffer inputIntData = intArrayToByteBuffer(randIntData);
                    tensor.setData(inputIntData);
                    break;
                }
                default: {
                    byte[] randByteData = generateByteArray(elementNums);
                    ByteBuffer inputByteData = byteArrayToByteBuffer(randByteData);
                    tensor.setData(inputByteData);
                    break;
                }
            }
        }

        // Run Inference.
        boolean ret = models.get(mdlIndex).predict();
        if (!ret) {
            freeTensorBuffer(tensors);
            System.err.println("MindSpore Lite run failed.");
            return false;
        }

        // Get first Output Tensor Data.
        MSTensor outTensor = models.get(mdlIndex).getOutputs().get(0);

        // Print out Tensor Data.
        StringBuilder msgSb = new StringBuilder();
        msgSb.append("out tensor shape: [");
        int[] shape = outTensor.getShape();
        for (int dim : shape) {
            msgSb.append(dim).append(",");
        }
        msgSb.append("]");
        int outDataType = outTensor.getDataType();
        if (outDataType != DataType.kNumberTypeFloat16) {
            freeTensorBuffer(tensors);
            outTensor.free();
            System.err.println("output tensor data type is not float16, the data type is " + outDataType);
            return false;
        }
        float[] result = outTensor.getFloatData();
        if (result == null) {
            freeTensorBuffer(tensors);
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
        freeTensorBuffer(tensors);
        outTensor.free();
        return true;
    }

    private static void freeModelBuffer() {
        for (Model m : models) {
            m.free();
        }
    }

    private static void freeTensorBuffer(List<MSTensor> tensors) {
        for (MSTensor t : tensors) {
            t.free();
        }
    }

    public static void main(String[] args) {
        System.out.println(Version.version());
        if (args.length < 5) {
            System.err.println("Please check input parameters.");
            return;
        }
        String modelPath = args[0];
        String incModelPath = args[1];
        int deviceId = Integer.parseInt(args[2]);
        int rankId = Integer.parseInt(args[3]);
        String config = args[4];
        boolean ret = compile(modelPath, incModelPath, deviceId, rankId, config);
        if (!ret) {
            System.err.println("MindSpore Lite compile failed.");
            return;
        }

        ret = run(false); // run main model
        if (!ret) {
            System.err.println("MindSpore Lite run main model failed.");
            freeModelBuffer();
            return;
        }
        ret = run(true); // run incremental model
        if (!ret) {
            System.err.println("MindSpore Lite run incremental model failed.");
            freeModelBuffer();
            return;
        }

        freeModelBuffer();
        System.out.println("End of distributed inference task!");
        return;
    }
}
