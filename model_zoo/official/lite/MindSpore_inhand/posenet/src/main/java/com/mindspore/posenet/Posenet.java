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
package com.mindspore.posenet;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import androidx.core.util.Pair;

import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.Model;
import com.mindspore.lite.config.CpuBindMode;
import com.mindspore.lite.config.DeviceType;
import com.mindspore.lite.config.MSConfig;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;

import static java.lang.Math.exp;

public class Posenet {

    public enum BodyPart {
        NOSE,
        LEFT_EYE,
        RIGHT_EYE,
        LEFT_EAR,
        RIGHT_EAR,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE
    }

    public class Position {
        float x;
        float y;
    }

    public class KeyPoint {
        BodyPart bodyPart = BodyPart.NOSE;
        Position position = new Position();
        float score = 0.0f;
    }

    public class Person {
        List<KeyPoint> keyPoints;
        float score = 0.0f;
    }

    private Context mContext;

    private MSConfig msConfig;
    private LiteSession session;
    private Model model;
    private LinkedHashMap<String, MSTensor> mOutputs;


    public long lastInferenceTimeNanos;
    private final int NUM_THREADS = 4;

    public Posenet(Context context) {
        mContext = context;
        init();
    }

    public boolean init() {
        // Load the .ms model.
        model = new Model();
        if (!model.loadModel(mContext, "posenet_model.ms")) {
            Log.e("MS_LITE", "Load Model failed");
            return false;
        }

        // Create and init config.
        msConfig = new MSConfig();
        if (!msConfig.init(DeviceType.DT_CPU, NUM_THREADS, CpuBindMode.MID_CPU)) {
            Log.e("MS_LITE", "Init context failed");
            return false;
        }

        // Create the MindSpore lite session.
        session = new LiteSession();
        if (!session.init(msConfig)) {
            Log.e("MS_LITE", "Create session failed");
            msConfig.free();
            return false;
        }
        msConfig.free();

        // Compile graph.
        if (!session.compileGraph(model)) {
            Log.e("MS_LITE", "Compile graph failed");
            model.freeBuffer();
            return false;
        }

        // Note: when use model.freeBuffer(), the model can not be compile graph again.
        model.freeBuffer();

        return true;
    }


    private float sigmoid(float x) {
        return (float) (1.0f / (1.0f + exp(-x)));
    }

    /**
     * Scale the image to a byteBuffer of [-1,1] values.
     */
    private ByteBuffer initInputArray(Bitmap bitmap) {
        final int bytesPerChannel = 4;
        final int inputChannels = 3;
        final int batchSize = 1;
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(
                batchSize * bytesPerChannel * bitmap.getHeight() * bitmap.getWidth() * inputChannels
        );
        inputBuffer.order(ByteOrder.nativeOrder());
        inputBuffer.rewind();

        final float mean = 128.0f;
        final float std = 128.0f;
        int[] intValues = new int[bitmap.getWidth() * bitmap.getHeight()];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int y = 0; y < bitmap.getHeight(); y++) {
            for (int x = 0; x < bitmap.getWidth(); x++) {
                int value = intValues[pixel++];
                inputBuffer.putFloat(((float) (value >> 16 & 0xFF) - mean) / std);
                inputBuffer.putFloat(((float) (value >> 8 & 0xFF) - mean) / std);
                inputBuffer.putFloat(((float) (value & 0xFF) - mean) / std);
            }
        }
        return inputBuffer;
    }


    /**
     * Estimates the pose for a single person.
     * args:
     * bitmap: image bitmap of frame that should be processed
     * returns:
     * person: a Person object containing data about keypoint locations and confidence scores
     */
    public Person estimateSinglePose(Bitmap bitmap) {
        long estimationStartTimeNanos = SystemClock.elapsedRealtimeNanos();
        ByteBuffer inputArray = this.initInputArray(bitmap);
        List<MSTensor> inputs = session.getInputs();
        if (inputs.size() != 1) {
            return null;
        }

        Log.i("posenet", String.format("Scaling to [-1,1] took %.2f ms",
                1.0f * (SystemClock.elapsedRealtimeNanos() - estimationStartTimeNanos) / 1_000_000));

        MSTensor inTensor = inputs.get(0);
        inTensor.setData(inputArray);
        long inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos();

        // Run graph to infer results.
        if (!session.runGraph()) {
            Log.e("MS_LITE", "Run graph failed");
            return null;
        }

        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos;
        Log.i(
                "posenet",
                String.format("Interpreter took %.2f ms", 1.0f * lastInferenceTimeNanos / 1_000_000)
        );

        float[][][][] heatmaps = runConv2Dfor27();
        float[][][][] offsets = runConv2Dfor28();

        if (heatmaps == null || offsets ==null){
            return null;
        }

        int height = ((Object[]) heatmaps[0]).length;  //9
        int width = ((Object[]) heatmaps[0][0]).length; //9
        int numKeypoints = heatmaps[0][0][0].length; //17

        // Finds the (row, col) locations of where the keypoints are most likely to be.
        Pair[] keypointPositions = new Pair[numKeypoints];
        for (int i = 0; i < numKeypoints; i++) {
            keypointPositions[i] = new Pair(0, 0);
        }

        for (int keypoint = 0; keypoint < numKeypoints; keypoint++) {
            float maxVal = heatmaps[0][0][0][keypoint];
            int maxRow = 0;
            int maxCol = 0;
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    if (heatmaps[0][row][col][keypoint] > maxVal) {
                        maxVal = heatmaps[0][row][col][keypoint];
                        maxRow = row;
                        maxCol = col;
                    }
                }
            }
            keypointPositions[keypoint] = new Pair(maxRow, maxCol);
        }

        // Calculating the x and y coordinates of the keypoints with offset adjustment.
        float[] xCoords = new float[numKeypoints];
        float[] yCoords = new float[numKeypoints];
        float[] confidenceScores = new float[numKeypoints];
        for (int i = 0; i < keypointPositions.length; i++) {
            Pair position = keypointPositions[i];
            int positionY = (int) position.first;
            int positionX = (int) position.second;

            yCoords[i] = ((float) positionY / (float) (height - 1) * bitmap.getHeight() + offsets[0][positionY][positionX][i]);
            xCoords[i] = ((float) positionX / (float) (width - 1) * bitmap.getWidth() + offsets[0][positionY][positionX][i + numKeypoints]);
            confidenceScores[i] = sigmoid(heatmaps[0][positionY][positionX][i]);
        }

        Person person = new Person();
        KeyPoint[] keypointList = new KeyPoint[numKeypoints];
        for (int i = 0; i < numKeypoints; i++) {
            keypointList[i] = new KeyPoint();
        }

        float totalScore = 0.0f;
        for (int i = 0; i < keypointList.length; i++) {
            keypointList[i].position.x = xCoords[i];
            keypointList[i].position.y = yCoords[i];
            keypointList[i].score = confidenceScores[i];
            totalScore += confidenceScores[i];
        }
        person.keyPoints = Arrays.asList(keypointList);
        person.score = totalScore / numKeypoints;

        return person;
    }

    private float[][][][] runConv2Dfor27() {
        // Get output tensor values.
        List<MSTensor> heatmaps_list = session.getOutputsByNodeName("Conv2D-27");
        if (heatmaps_list == null) {
            return null;
        }
        MSTensor heatmaps_tensors = heatmaps_list.get(0);

        float[] heatmaps_results = heatmaps_tensors.getFloatData();
        int[] heatmapsShape = heatmaps_tensors.getShape(); //1, 9, 9 ,17

        if (heatmapsShape[0] < 0 || heatmapsShape[1] < 0 || heatmapsShape[2] < 0 || heatmapsShape[3] < 0) {
            return null;
        }

        float[][][][] heatmaps = new float[heatmapsShape[0]][][][];
        for (int x = 0; x < heatmapsShape[0]; x++) {  // heatmapsShape[0] =1
            float[][][] arrayThree = new float[heatmapsShape[1]][][];
            for (int y = 0; y < heatmapsShape[1]; y++) {  // heatmapsShape[1] = 9
                float[][] arrayTwo = new float[heatmapsShape[2]][];
                for (int z = 0; z < heatmapsShape[2]; z++) { //heatmapsShape[2] = 9
                    float[] arrayOne = new float[heatmapsShape[3]]; //heatmapsShape[3] = 17
                    for (int i = 0; i < heatmapsShape[3]; i++) {
                        int n = i + z * heatmapsShape[3] + y * heatmapsShape[2] * heatmapsShape[3] + x * heatmapsShape[1] * heatmapsShape[2] * heatmapsShape[3];
                        arrayOne[i] = heatmaps_results[n]; //1*9*9*17  ??
                    }
                    arrayTwo[z] = arrayOne;
                }
                arrayThree[y] = arrayTwo;
            }
            heatmaps[x] = arrayThree;
        }
        return heatmaps;
    }


    private float[][][][] runConv2Dfor28() {
        List<MSTensor> offsets_list = session.getOutputsByNodeName("Conv2D-28");
        if (offsets_list == null) {
            return null;
        }
        MSTensor offsets_tensors = offsets_list.get(0);
        float[] offsets_results = offsets_tensors.getFloatData();
        int[] offsetsShapes = offsets_tensors.getShape();

        if (offsetsShapes[0] < 0 || offsetsShapes[1] < 0 || offsetsShapes[2] < 0 || offsetsShapes[3] < 0) {
            return null;
        }
        float[][][][] offsets = new float[offsetsShapes[0]][][][];
        for (int x = 0; x < offsetsShapes[0]; x++) {
            float[][][] offsets_arrayThree = new float[offsetsShapes[1]][][];
            for (int y = 0; y < offsetsShapes[1]; y++) {
                float[][] offsets_arrayTwo = new float[offsetsShapes[2]][];
                for (int z = 0; z < offsetsShapes[2]; z++) {
                    float[] offsets_arrayOne = new float[offsetsShapes[3]];
                    for (int i = 0; i < offsetsShapes[3]; i++) {
                        int n = i + z * offsetsShapes[3] + y * offsetsShapes[2] * offsetsShapes[3] + x * offsetsShapes[1] * offsetsShapes[2] * offsetsShapes[3];
                        offsets_arrayOne[i] = offsets_results[n];
                    }
                    offsets_arrayTwo[z] = offsets_arrayOne;
                }
                offsets_arrayThree[y] = offsets_arrayTwo;
            }
            offsets[x] = offsets_arrayThree;
        }
        return offsets;
    }
}
