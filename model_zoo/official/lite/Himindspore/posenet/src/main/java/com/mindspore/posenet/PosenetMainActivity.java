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

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.hardware.camera2.CameraCharacteristics;
import android.media.Image;
import android.os.Bundle;
import android.view.SurfaceView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.util.Pair;

import com.alibaba.android.arouter.facade.annotation.Route;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

import static com.mindspore.posenet.Posenet.BodyPart.LEFT_ANKLE;
import static com.mindspore.posenet.Posenet.BodyPart.LEFT_ELBOW;
import static com.mindspore.posenet.Posenet.BodyPart.LEFT_HIP;
import static com.mindspore.posenet.Posenet.BodyPart.LEFT_KNEE;
import static com.mindspore.posenet.Posenet.BodyPart.LEFT_SHOULDER;
import static com.mindspore.posenet.Posenet.BodyPart.LEFT_WRIST;
import static com.mindspore.posenet.Posenet.BodyPart.RIGHT_ANKLE;
import static com.mindspore.posenet.Posenet.BodyPart.RIGHT_ELBOW;
import static com.mindspore.posenet.Posenet.BodyPart.RIGHT_HIP;
import static com.mindspore.posenet.Posenet.BodyPart.RIGHT_KNEE;
import static com.mindspore.posenet.Posenet.BodyPart.RIGHT_SHOULDER;
import static com.mindspore.posenet.Posenet.BodyPart.RIGHT_WRIST;

@Route(path = "/posenet/PosenetMainActivity")
public class PosenetMainActivity extends AppCompatActivity implements CameraDataDealListener {

    private final List bodyJoints;


    /**
     * Model input shape for images.
     */
    private final static int MODEL_WIDTH = 257;
    private final static int MODEL_HEIGHT = 257;

    private final double minConfidence = 0.5;
    private final float circleRadius = 8.0f;
    private Paint paint = new Paint();
    private final int PREVIEW_WIDTH = 640;
    private final int PREVIEW_HEIGHT = 480;
    private Posenet posenet;
    private int[] rgbBytes = new int[PREVIEW_WIDTH * PREVIEW_HEIGHT];
    private byte[][] yuvBytes = new byte[3][];
    private SurfaceView surfaceView;

    private int lensFacing = CameraCharacteristics.LENS_FACING_BACK;
    private PoseNetFragment poseNetFragment;

    public PosenetMainActivity() {
        bodyJoints = Arrays.asList(
                new Pair(LEFT_WRIST, LEFT_ELBOW), new Pair(LEFT_ELBOW, LEFT_SHOULDER),
                new Pair(LEFT_SHOULDER, RIGHT_SHOULDER), new Pair(RIGHT_SHOULDER, RIGHT_ELBOW),
                new Pair(RIGHT_ELBOW, RIGHT_WRIST), new Pair(LEFT_SHOULDER, LEFT_HIP),
                new Pair(LEFT_HIP, RIGHT_HIP), new Pair(RIGHT_HIP, RIGHT_SHOULDER),
                new Pair(LEFT_HIP, LEFT_KNEE), new Pair(LEFT_KNEE, LEFT_ANKLE),
                new Pair(RIGHT_HIP, RIGHT_KNEE), new Pair(RIGHT_KNEE, RIGHT_ANKLE));
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        addCameraFragment();
    }

    private void addCameraFragment() {
        posenet = new Posenet(this);

        poseNetFragment = PoseNetFragment.newInstance();
        poseNetFragment.setCameraDataDealListener(this);
        //   poseNetFragment.setFacingCamera(lensFacing);
        getSupportFragmentManager().popBackStack();
        getSupportFragmentManager().beginTransaction()
                .replace(R.id.container, poseNetFragment)
                .commitAllowingStateLoss();
    }

    @Override
    public void dataDeal(Image image, SurfaceView surfaceView) {
        if (image == null || image.getPlanes() == null) {
            return;
        }
        this.surfaceView = surfaceView;
        fillBytes(image.getPlanes(), yuvBytes);
        ImageUtils.convertYUV420ToARGB8888(yuvBytes[0], yuvBytes[1], yuvBytes[2],
                PREVIEW_WIDTH, PREVIEW_HEIGHT,
                image.getPlanes()[0].getRowStride(),
                image.getPlanes()[1].getRowStride(),
                image.getPlanes()[1].getPixelStride(),
                rgbBytes);

        Bitmap imageBitmap = Bitmap.createBitmap(
                rgbBytes, PREVIEW_WIDTH, PREVIEW_HEIGHT,
                Bitmap.Config.ARGB_8888);
        Matrix rotateMatrix = new Matrix();
        rotateMatrix.postRotate(90.0f);

        Bitmap rotatedBitmap = Bitmap.createBitmap(
                imageBitmap, 0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT,
                rotateMatrix, true
        );
        image.close();
        processImage(rotatedBitmap);
    }


    /**
     * Fill the yuvBytes with data from image planes.
     */
    private void fillBytes(Image.Plane[] planes, byte[][] yuvBytes) {
        // Row stride is the total number of bytes occupied in memory by a row of an image.
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes
        for (int i = 0; i < planes.length; ++i) {
            ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    /**
     * Crop Bitmap to maintain aspect ratio of model input.
     */
    private Bitmap cropBitmap(Bitmap bitmap) {
        float bitmapRatio = bitmap.getHeight() / bitmap.getWidth();
        float modelInputRatio = MODEL_HEIGHT / MODEL_WIDTH;
        double maxDifference = 1.0E-5D;
        float cropHeight = modelInputRatio - bitmapRatio;

        if (Math.abs(cropHeight) < maxDifference) {
            return bitmap;
        } else {
            Bitmap croppedBitmap;
            if (modelInputRatio < bitmapRatio) {
                cropHeight = (float) bitmap.getHeight() - (float) bitmap.getWidth() / modelInputRatio;
                croppedBitmap = Bitmap.createBitmap(bitmap,
                        0, (int) (cropHeight / 2), bitmap.getWidth(), (int) (bitmap.getHeight() - cropHeight));
            } else {
                cropHeight = (float) bitmap.getWidth() - (float) bitmap.getHeight() * modelInputRatio;
                croppedBitmap = Bitmap.createBitmap(bitmap,
                        (int) (cropHeight / 2), 0, (int) (bitmap.getWidth() - cropHeight), bitmap.getHeight());
            }
            return croppedBitmap;
        }
    }

    /**
     * Set the paint color and size.
     */
    private void setPaint() {
        paint.setColor(getResources().getColor(R.color.text_blue));
        paint.setTextSize(80.0f);
        paint.setStrokeWidth(8.0f);
    }

    /**
     * Draw bitmap on Canvas.
     */
    private void draw(Canvas canvas, Posenet.Person person, Bitmap bitmap) {
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
        // Draw `bitmap` and `person` in square canvas.
        int screenWidth, screenHeight;
        int left, right, top, bottom;
        if (canvas.getHeight() > canvas.getWidth()) {
            screenWidth = canvas.getWidth();
            screenHeight = canvas.getWidth();
            left = 0;
            top = (canvas.getHeight() - canvas.getWidth()) / 2;
        } else {
            screenWidth = canvas.getHeight();
            screenHeight = canvas.getHeight();
            left = (canvas.getWidth() - canvas.getHeight()) / 2;
            top = 0;
        }
        right = left + screenWidth;
        bottom = top + screenHeight;

        setPaint();
        canvas.drawBitmap(
                bitmap,
                new Rect(0, 0, bitmap.getWidth(), bitmap.getHeight()),
                new Rect(left, top, right, bottom), paint);

        float widthRatio = screenWidth / MODEL_WIDTH;
        float heightRatio = screenHeight / MODEL_HEIGHT;

        for (Posenet.KeyPoint keyPoint : person.keyPoints) {
            if (keyPoint.score > minConfidence) {
                Posenet.Position position = keyPoint.position;
                float adjustedX = position.x * widthRatio + left;
                float adjustedY = position.y * heightRatio + top;
                canvas.drawCircle(adjustedX, adjustedY, circleRadius, paint);
            }
        }

        for (int i = 0; i < bodyJoints.size(); i++) {
            Pair line = (Pair) bodyJoints.get(i);
            Posenet.BodyPart first = (Posenet.BodyPart) line.first;
            Posenet.BodyPart second = (Posenet.BodyPart) line.second;

            if (person.keyPoints.get(first.ordinal()).score > minConfidence &
                    person.keyPoints.get(second.ordinal()).score > minConfidence) {
                canvas.drawLine(
                        person.keyPoints.get(first.ordinal()).position.x * widthRatio + left,
                        person.keyPoints.get(first.ordinal()).position.y * heightRatio + top,
                        person.keyPoints.get(second.ordinal()).position.x * widthRatio + left,
                        person.keyPoints.get(second.ordinal()).position.y * heightRatio + top, paint);
            }
        }

        canvas.drawText(String.format("Score: %.2f", person.score),
                (15.0f * widthRatio), (30.0f * heightRatio + bottom), paint);
        canvas.drawText(String.format("Time: %.2f ms", posenet.lastInferenceTimeNanos * 1.0f / 1_000_000),
                (15.0f * widthRatio), (50.0f * heightRatio + bottom), paint
        );

        // Draw!
        surfaceView.getHolder().unlockCanvasAndPost(canvas);
    }

    /**
     * Process image using Posenet library.
     */
    private void processImage(Bitmap bitmap) {
        // Crop bitmap.
        Bitmap croppedBitmap = cropBitmap(bitmap);
        // Created scaled version of bitmap for model input.
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, MODEL_WIDTH, MODEL_HEIGHT, true);
        // Perform inference.
        Posenet.Person person = posenet.estimateSinglePose(scaledBitmap);
        Canvas canvas = surfaceView.getHolder().lockCanvas();
        draw(canvas, person, scaledBitmap);
    }

}