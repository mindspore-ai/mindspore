/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.styletransfer;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.Model;
import com.mindspore.lite.config.CpuBindMode;
import com.mindspore.lite.config.DeviceType;
import com.mindspore.lite.config.MSConfig;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class StyleTransferModelExecutor {

    private static final String TAG = "StyleTransferModelExecutor";
    private static final int STYLE_IMAGE_SIZE = 256;
    private static final int CONTENT_IMAGE_SIZE = 384;

    private Context mContext;

    private MSConfig msConfig;
    private LiteSession Predict_session;
    private LiteSession Transform_session;
    private Model style_predict_model;
    private Model style_transform_model;
    private LinkedHashMap<String, MSTensor> mOutputs;

    private long fullExecutionTime;
    private long preProcessTime;
    private long stylePredictTime;
    private long styleTransferTime;
    private long postProcessTime;

    private final int NUM_THREADS = 4;

    public StyleTransferModelExecutor(Context context, boolean useGPU) {
        mContext = context;
        init();
    }

    public void init() {
        // Load the .ms model.
        style_predict_model = new Model();
        if (!style_predict_model.loadModel(mContext, "style_predict_quant.ms")) {
            Log.e("MS_LITE", "Load style_predict_model failed");
        }

        style_transform_model = new Model();
        if (!style_transform_model.loadModel(mContext, "style_transfer_quant.ms")) {
            Log.e("MS_LITE", "Load style_transform_model failed");
        }
        // Create and init config.
        msConfig = new MSConfig();
        if (!msConfig.init(DeviceType.DT_CPU, NUM_THREADS, CpuBindMode.MID_CPU)) {
            Log.e("MS_LITE", "Init context failed");
        }

        // Create the MindSpore lite session.
        Predict_session = new LiteSession();
        if (!Predict_session.init(msConfig)) {
            Log.e("MS_LITE", "Create Predict_session failed");
            msConfig.free();
        }

        Transform_session = new LiteSession();
        if (!Transform_session.init(msConfig)) {
            Log.e("MS_LITE", "Create Predict_session failed");
            msConfig.free();
        }
        msConfig.free();


        // Compile graph.
        if (!Predict_session.compileGraph(style_predict_model)) {
            Log.e("MS_LITE", "Compile style_predict graph failed");
            style_predict_model.freeBuffer();
        }
        if (!Transform_session.compileGraph(style_transform_model)) {
            Log.e("MS_LITE", "Compile style_transform graph failed");
            style_transform_model.freeBuffer();
        }

        // Note: when use model.freeBuffer(), the model can not be compile graph again.
        style_predict_model.freeBuffer();
        style_transform_model.freeBuffer();
    }


    /**
     * @param floats the floats
     * @return the byte [ ]
     */
    public static byte[] floatArrayToByteArray(float[] floats) {
        ByteBuffer buffer = ByteBuffer.allocate(4 * floats.length);
        buffer.order(ByteOrder.nativeOrder());
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.put(floats);
        return buffer.array();
    }

    @SuppressLint("LongLogTag")
    public ModelExecutionResult execute(Bitmap contentImage, Bitmap styleBitmap) {
        Log.i(TAG, "running models");

        fullExecutionTime = SystemClock.uptimeMillis();
        preProcessTime = SystemClock.uptimeMillis();
        ByteBuffer contentArray =
                ImageUtils.bitmapToByteBuffer(contentImage, CONTENT_IMAGE_SIZE, CONTENT_IMAGE_SIZE, 0, 255);
        ByteBuffer input = ImageUtils.bitmapToByteBuffer(styleBitmap, STYLE_IMAGE_SIZE, STYLE_IMAGE_SIZE, 0, 255);

        List<MSTensor> Predict_inputs = Predict_session.getInputs();
        if (Predict_inputs.size() != 1) {
            return null;
        }
        MSTensor Predict_inTensor = Predict_inputs.get(0);
        Predict_inTensor.setData(input);

        preProcessTime = SystemClock.uptimeMillis() - preProcessTime;
        stylePredictTime = SystemClock.uptimeMillis();

        if (!Predict_session.runGraph()) {
            Log.e("MS_LITE", "Run Predict_graph failed");
            return null;
        }
        stylePredictTime = SystemClock.uptimeMillis() - stylePredictTime;
        Log.d(TAG, "Style Predict Time to run: " + stylePredictTime);

        float[][][][] outputImage = getDealData(contentArray);

        Bitmap styledImage =
                ImageUtils.convertArrayToBitmap(outputImage, CONTENT_IMAGE_SIZE, CONTENT_IMAGE_SIZE);
        postProcessTime = SystemClock.uptimeMillis() - postProcessTime;

        fullExecutionTime = SystemClock.uptimeMillis() - fullExecutionTime;
        Log.d(TAG, "Time to run everything: $" + fullExecutionTime);

        return new ModelExecutionResult(styledImage,
                preProcessTime,
                stylePredictTime,
                styleTransferTime,
                postProcessTime,
                fullExecutionTime,
                formatExecutionLog());
    }

    @SuppressLint("LongLogTag")
    private float[][][][] getDealData(ByteBuffer contentArray) {
        // Get output tensor values.
        List<String> tensorNames = Predict_session.getOutputTensorNames();
        Map<String, MSTensor> outputs = Predict_session.getOutputMapByTensor();

        float[] Predict_results = null;
        for (String tensorName : tensorNames) {
            MSTensor output = outputs.get(tensorName);
            if (output == null) {
                Log.e("MS_LITE", "Can not find Predict_session output " + tensorName);
                return null;
            }
            Predict_results = output.getFloatData();
        }

        List<MSTensor> Transform_inputs = Transform_session.getInputs();
        // transform model have 2 input tensor,  tensor0: 1*1*1*100,   tensor1；1*384*384*3
        MSTensor Transform_inputs_inTensor0 = Transform_inputs.get(0);
        Transform_inputs_inTensor0.setData(floatArrayToByteArray(Predict_results));

        MSTensor Transform_inputs_inTensor1 = Transform_inputs.get(1);
        Transform_inputs_inTensor1.setData(contentArray);
        styleTransferTime = SystemClock.uptimeMillis();

        if (!Transform_session.runGraph()) {
            Log.e("MS_LITE", "Run Transform_graph failed");
            return null;
        }

        styleTransferTime = SystemClock.uptimeMillis() - styleTransferTime;
        Log.d(TAG, "Style apply Time to run: " + styleTransferTime);
        postProcessTime = SystemClock.uptimeMillis();

        // Get output tensor values.
        List<String> Transform_tensorNames = Transform_session.getOutputTensorNames();
        Map<String, MSTensor> Transform_outputs = Transform_session.getOutputMapByTensor();

        float[] transform_results = null;
        for (String tensorName : Transform_tensorNames) {
            MSTensor output1 = Transform_outputs.get(tensorName);
            if (output1 == null) {
                Log.e("MS_LITE", "Can not find Transform_session output " + tensorName);
                return null;
            }
            transform_results = output1.getFloatData();
        }

        float[][][][] outputImage = new float[1][][][];  // 1 384 384 3
        for (int x = 0; x < 1; x++) {
            float[][][] arrayThree = new float[CONTENT_IMAGE_SIZE][][];
            for (int y = 0; y < CONTENT_IMAGE_SIZE; y++) {
                float[][] arrayTwo = new float[CONTENT_IMAGE_SIZE][];
                for (int z = 0; z < CONTENT_IMAGE_SIZE; z++) {
                    float[] arrayOne = new float[3];
                    for (int i = 0; i < 3; i++) {
                        int n = i + z * 3 + y * CONTENT_IMAGE_SIZE * 3 + x * CONTENT_IMAGE_SIZE * CONTENT_IMAGE_SIZE * 3;
                        arrayOne[i] = transform_results[n];
                    }
                    arrayTwo[z] = arrayOne;
                }
                arrayThree[y] = arrayTwo;
            }
            outputImage[x] = arrayThree;
        }
        return outputImage;
    }

    private String formatExecutionLog() {
        StringBuilder sb = new StringBuilder();
        sb.append("Input Image Size:" + CONTENT_IMAGE_SIZE * CONTENT_IMAGE_SIZE)
                .append("\nPre-process execution time: " + preProcessTime + " ms")
                .append("\nPredicting style execution time: " + stylePredictTime + " ms")
                .append("\nTransferring style execution time: " + styleTransferTime + " ms")
                .append("\nPost-process execution time: " + postProcessTime + " ms")
                .append("\nFull execution time: " + fullExecutionTime + " ms");
        return sb.toString();
    }

}
