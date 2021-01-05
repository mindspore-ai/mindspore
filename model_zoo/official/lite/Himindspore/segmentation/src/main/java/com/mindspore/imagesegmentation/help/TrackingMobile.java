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
package com.mindspore.imagesegmentation.help;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import androidx.core.graphics.ColorUtils;

import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.Model;
import com.mindspore.lite.config.CpuBindMode;
import com.mindspore.lite.config.DeviceType;
import com.mindspore.lite.config.MSConfig;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class TrackingMobile {
    private static final String TAG = "TrackingMobile";

    private static final String IMAGESEGMENTATIONMODEL = "segment_model.ms";
    private static final int imageSize = 257;
    public static final int NUM_CLASSES = 21;
    private static final float IMAGE_MEAN = 127.5F;
    private static final float IMAGE_STD = 127.5F;

    public static final int[] segmentColors = new int[2];

    private Bitmap maskBitmap;
    private Bitmap resultBitmap;
    private HashSet itemsFound = new HashSet();

    private final Context mContext;

    private MSConfig msConfig;
    private LiteSession session;
    private Model model;

    public TrackingMobile(Context context) {
        mContext = context;
        init();
    }

    public void init() {
        // Load the .ms model.
        model = new Model();
        if (!model.loadModel(mContext, IMAGESEGMENTATIONMODEL)) {
            Log.e(TAG, "Load Model failed");
            return;
        }

        // Create and init config.
        msConfig = new MSConfig();
        if (!msConfig.init(DeviceType.DT_CPU, 2, CpuBindMode.MID_CPU)) {
            Log.e(TAG, "Init context failed");
            return;
        }

        // Create the MindSpore lite session.
        session = new LiteSession();
        if (!session.init(msConfig)) {
            Log.e(TAG, "Create session failed");
            msConfig.free();
            return;
        }
        msConfig.free();

        // Compile graph.
        if (!session.compileGraph(model)) {
            Log.e(TAG, "Compile graph failed");
            model.freeBuffer();
            return;
        }

        // Note: when use model.freeBuffer(), the model can not be compile graph again.
        model.freeBuffer();

    }

    public ModelTrackingResult execute(Bitmap bitmap) {
        // Set input tensor values.
        List<MSTensor> inputs = session.getInputs();
        if (inputs.size() != 1) {
            Log.e(TAG, "inputs.size() != 1");
            return null;
        }

        float resource_height = bitmap.getHeight();
        float resource_weight = bitmap.getWidth();

        Bitmap scaledBitmap = BitmapUtils.scaleBitmapAndKeepRatio(bitmap, imageSize, imageSize);
        ByteBuffer contentArray = BitmapUtils.bitmapToByteBuffer(scaledBitmap, imageSize, imageSize, IMAGE_MEAN, IMAGE_STD);

        MSTensor inTensor = inputs.get(0);
        //    int byteLen = (int) inTensor.size();
        inTensor.setData(contentArray);

        // Run graph to infer results.
        if (!session.runGraph()) {
            Log.e(TAG, "Run graph failed");
            return null;
        }

        // Get output tensor values.
        List<String> tensorNames = session.getOutputTensorNames();
        Map<String, MSTensor> outputs = session.getOutputMapByTensor();
        for (String tensorName : tensorNames) {
            MSTensor output = outputs.get(tensorName);
            if (output == null) {
                Log.e(TAG, "Can not find output " + tensorName);
                return null;
            }
            float[] results = output.getFloatData();
            float[] result = new float[output.elementsNum()];

            int batch = output.getShape()[0];
            int channel = output.getShape()[1];
            int weight = output.getShape()[2];
            int height = output.getShape()[3];
            int plane = weight * height;

            for (int n = 0; n < batch; n++) {
                for (int c = 0; c < channel; c++) {
                    for (int hw = 0; hw < plane; hw++) {
                        result[n * channel * plane + hw * channel + c] = results[n * channel * plane + c * plane + hw];
                    }
                }
            }
            ByteBuffer bytebuffer_results = floatArrayToByteArray(result);

            convertBytebufferMaskToBitmap(
                    bytebuffer_results, imageSize, imageSize, scaledBitmap,
                    segmentColors
            );
            //scaledBitmap resize成resource_height，resource_weight
            scaledBitmap = BitmapUtils.scaleBitmapAndKeepRatio(scaledBitmap, (int) resource_height, (int) resource_weight);
            resultBitmap = BitmapUtils.scaleBitmapAndKeepRatio(resultBitmap, (int) resource_height, (int) resource_weight);
            maskBitmap = BitmapUtils.scaleBitmapAndKeepRatio(maskBitmap, (int) resource_height, (int) resource_weight);
        }
        return new ModelTrackingResult(resultBitmap, scaledBitmap, maskBitmap, this.formatExecutionLog(), itemsFound);
    }

    private static ByteBuffer floatArrayToByteArray(float[] floats) {
        ByteBuffer buffer = ByteBuffer.allocate(4 * floats.length);
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.put(floats);
        return buffer;
    }

    private void convertBytebufferMaskToBitmap(ByteBuffer inputBuffer, int imageWidth,
                                               int imageHeight, Bitmap backgroundImage, int[] colors) {
        Bitmap.Config conf = Bitmap.Config.ARGB_8888;
        maskBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf);
        resultBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf);
        Bitmap scaledBackgroundImage =
                BitmapUtils.scaleBitmapAndKeepRatio(backgroundImage, imageWidth, imageHeight);
        int[][] mSegmentBits = new int[imageWidth][imageHeight];
        inputBuffer.rewind();
        for (int y = 0; y < imageHeight; y++) {
            for (int x = 0; x < imageWidth; x++) {
                float maxVal = 0f;
                mSegmentBits[x][y] = 0;
                for (int i = 0; i < NUM_CLASSES; i++) {
                    float value = inputBuffer.getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + i) * 4);
                    if (i == 0 || value > maxVal) {
                        maxVal = value;
                        if (i == 15) {
                            mSegmentBits[x][y] = i;
                        } else {
                            mSegmentBits[x][y] = 0;
                        }
                    }
                }
                itemsFound.add(mSegmentBits[x][y]);

                int newPixelColor = ColorUtils.compositeColors(
                        colors[mSegmentBits[x][y] == 0 ? 0 : 1],
                        scaledBackgroundImage.getPixel(x, y)
                );
                resultBitmap.setPixel(x, y, newPixelColor);
                maskBitmap.setPixel(x, y, mSegmentBits[x][y] == 0 ? colors[0] : scaledBackgroundImage.getPixel(x, y));
            }
        }
    }

    // Note: we must release the memory at the end, otherwise it will cause the memory leak.
    public void free() {
        session.free();
        model.free();
    }


    private final String formatExecutionLog() {
        StringBuilder sb = new StringBuilder();
        sb.append("Input Image Size: " + imageSize * imageSize + '\n');
        return sb.toString();
    }

}
