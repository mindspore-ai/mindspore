package com.mindspore.lite.demo;

import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.Model;
import com.mindspore.lite.DataType;
import com.mindspore.lite.Version;
import com.mindspore.lite.config.MSConfig;
import com.mindspore.lite.config.DeviceType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Random;

public class Main {
    private static Model model;
    private static LiteSession session;

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

    private static boolean compile() {
        MSConfig msConfig = new MSConfig();
        // You can set config through Init Api or use the default parameters directly.
        // The default parameter is that the backend type is DeviceType.DT_CPU, and the number of threads is 2.
        boolean ret = msConfig.init(DeviceType.DT_CPU, 2);
        if (!ret) {
            System.err.println("Init context failed");
            return false;
        }

        // Create the MindSpore lite session.
        session = new LiteSession();
        ret = session.init(msConfig);
        msConfig.free();
        if (!ret) {
            System.err.println("Create session failed");
            model.free();
            return false;
        }

        // Compile graph.
        ret = session.compileGraph(model);
        if (!ret) {
            System.err.println("Compile graph failed");
            model.free();
            return false;
        }
        return true;
    }

    private static boolean run() {
        MSTensor inputTensor = session.getInputsByTensorName("2031_2030_1_construct_wrapper:x");
        if (inputTensor.getDataType() != DataType.kNumberTypeFloat32) {
            System.err.println("Input tensor shape do not float, the data type is " + inputTensor.getDataType());
            return false;
        }
        // Generator Random Data.
        int elementNums = inputTensor.elementsNum();
        float[] randomData = generateArray(elementNums);
        ByteBuffer inputData = floatArrayToByteBuffer(randomData);

        // Set Input Data.
        inputTensor.setData(inputData);

        // Run Inference.
        boolean ret = session.runGraph();
        if (!ret) {
            System.err.println("MindSpore Lite run failed.");
            return false;
        }

        // Get Output Tensor Data.
        MSTensor outTensor = session.getOutputByTensorName("Default/head-MobileNetV2Head/Softmax-op204");

        // Print out Tensor Data.
        StringBuilder msgSb = new StringBuilder();
        msgSb.append("out tensor shape: [");
        int[] shape = outTensor.getShape();
        for (int dim : shape) {
            msgSb.append(dim).append(",");
        }
        msgSb.append("]");
        if (outTensor.getDataType() != DataType.kNumberTypeFloat32) {
            System.err.println("output tensor shape do not float, the data type is " + outTensor.getDataType());
            return false;
        }
        float[] result = outTensor.getFloatData();
        if (result == null) {
            System.err.println("decodeBytes return null");
            return false;
        }
        msgSb.append(" and out data:");
        for (int i = 0; i < 50 && i < outTensor.elementsNum(); i++) {
            msgSb.append(" ").append(result[i]);
        }
        System.out.println(msgSb.toString());
        return true;
    }

    private static void freeBuffer() {
        session.free();
        model.free();
    }

    public static void main(String[] args) {
        System.out.println(Version.version());
        if (args.length < 1) {
            System.err.println("The model path parameter must be passed.");
            return;
        }
        String modelPath = args[0];
        model = new Model();

        boolean ret = model.loadModel(modelPath);
        if (!ret) {
            System.err.println("Load model failed, model path is " + modelPath);
            return;
        }
        ret = compile();
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
