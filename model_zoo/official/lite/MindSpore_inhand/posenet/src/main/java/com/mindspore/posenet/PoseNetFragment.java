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

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.util.Pair;
import androidx.fragment.app.Fragment;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

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

/**
 * A simple {@link Fragment} subclass.
 * create an instance of this fragment.g
 */
public class PoseNetFragment extends Fragment {

    private final List bodyJoints = Arrays.asList(
            new Pair(LEFT_WRIST, LEFT_ELBOW), new Pair(LEFT_ELBOW, LEFT_SHOULDER),
            new Pair(LEFT_SHOULDER, RIGHT_SHOULDER), new Pair(RIGHT_SHOULDER, RIGHT_ELBOW),
            new Pair(RIGHT_ELBOW, RIGHT_WRIST), new Pair(LEFT_SHOULDER, LEFT_HIP),
            new Pair(LEFT_HIP, RIGHT_HIP), new Pair(RIGHT_HIP, RIGHT_SHOULDER),
            new Pair(LEFT_HIP, LEFT_KNEE), new Pair(LEFT_KNEE, LEFT_ANKLE),
            new Pair(RIGHT_HIP, RIGHT_KNEE), new Pair(RIGHT_KNEE, RIGHT_ANKLE));

    private static final String TAG = "PoseNetFragment";

    private int mCameraId = CameraCharacteristics.LENS_FACING_FRONT; // 要打开的摄像头ID
    private SurfaceView surfaceView;
    private CameraCaptureSession captureSession;
    private CameraDevice cameraDevice;
    private final static int PREVIEW_WIDTH = 640;
    private final static int PREVIEW_HEIGHT = 480;
    private Size previewSize = new Size(PREVIEW_WIDTH, PREVIEW_HEIGHT);
    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    private ImageReader imageReader;
    private CaptureRequest.Builder previewRequestBuilder;
    private CaptureRequest previewRequest;
    private Semaphore cameraOpenCloseLock = new Semaphore(1);//使用信号量 Semaphore 进行多线程任务调度
    private boolean flashSupported;
    private boolean isPreBackgroundThreadPause;

    /**
     * Model input shape for images.
     */
    private final static int MODEL_WIDTH = 257;
    private final static int MODEL_HEIGHT = 257;

    private final double minConfidence = 0.5;
    private final float circleRadius = 8.0f;
    private Paint paint = new Paint();
    private Posenet posenet;
    private int[] rgbBytes = new int[PREVIEW_WIDTH * PREVIEW_HEIGHT];
    private byte[][] yuvBytes = new byte[3][];

    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {

        @Override
        public void onOpened(@NonNull CameraDevice mCameraDevice) {
            cameraOpenCloseLock.release();
            Log.d(TAG, "camera has open");
            PoseNetFragment.this.cameraDevice = mCameraDevice;
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            cameraOpenCloseLock.release();
            closeCamera();
            PoseNetFragment.this.cameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            onDisconnected(cameraDevice);
            closeCamera();
            Activity activity = getActivity();
            if (activity != null) {
                activity.finish();
            }
        }
    };

    private CameraCaptureSession.CaptureCallback captureCallback = new CameraCaptureSession.CaptureCallback() {
        @Override
        public void onCaptureProgressed(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull CaptureResult partialResult) {
            super.onCaptureProgressed(session, request, partialResult);
        }

        @Override
        public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull TotalCaptureResult result) {
            super.onCaptureCompleted(session, request, result);
        }
    };

    public static PoseNetFragment newInstance() {
        PoseNetFragment fragment = new PoseNetFragment();
        return fragment;
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.posenet_fragment_pose_net, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        this.surfaceView = view.findViewById(R.id.surfaceView);
    }

    @Override
    public void onResume() {
        super.onResume();
        startBackgroundThread();
    }

    public void onStart() {
        super.onStart();
        openCamera();
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        posenet = new Posenet(getActivity());

    }

    public void onPause() {
        this.stopBackgroundThread();
        this.closeCamera();
        super.onPause();
    }

    public void switchCamera() {
        mCameraId ^= 1;
        Log.d(TAG, "switchCamera: mCameraId: " + mCameraId);
        closeCamera();
        openCamera();
    }

    /**
     * Opens the camera specified by [PosenetActivity.cameraId].
     */
    @SuppressLint("MissingPermission")
    private void openCamera() {
        CameraManager manager = (CameraManager) getContext().getSystemService(Context.CAMERA_SERVICE);
        try {
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(Integer.toString(mCameraId));
            previewSize = new Size(PREVIEW_WIDTH, PREVIEW_HEIGHT);
            imageReader = ImageReader.newInstance(
                    PREVIEW_WIDTH, PREVIEW_HEIGHT,
                    ImageFormat.YUV_420_888, /*maxImages*/ 2
            );
            imageReader.setOnImageAvailableListener(imageAvailableListener, backgroundHandler);
            // Check if the flash is supported.
            flashSupported =
                    characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true;
            // Wait for camera to open - 2.5 seconds is sufficient
            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            manager.openCamera(Integer.toString(mCameraId), mStateCallback, backgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void closeCamera() {
        Log.e("AAA", "closeCamera");
        try {
            cameraOpenCloseLock.acquire();
            if (captureSession != null) {
                captureSession.close();
                captureSession = null;
            }
            if (null != cameraDevice) {
                cameraDevice.close();
                cameraDevice = null;
            }
            if (null != imageReader) {
                imageReader.close();
                imageReader = null;
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
        } finally {
            cameraOpenCloseLock.release();
        }
    }

    /**
     * Starts a background thread and its [Handler].
     */
    private void startBackgroundThread() {
        Log.e("AAA", "startBackgroundThread");

        backgroundThread = new HandlerThread("imageAvailableListener");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    /**
     * Stops the background thread and its [Handler].
     */
    private void stopBackgroundThread() {
        Log.e("AAA", "stopBackgroundThread");
        isPreBackgroundThreadPause = true;
        backgroundThread.quitSafely();
        backgroundThread.interrupt();
        backgroundThread = null;
        backgroundHandler = null;

    }

    private final ImageReader.OnImageAvailableListener imageAvailableListener = (ImageReader mImageReader) -> {
        if (backgroundHandler != null && backgroundThread != null && !isPreBackgroundThreadPause) {
            if (mImageReader != null) {
                Image image = mImageReader.acquireLatestImage();
                if (image == null || image.getPlanes() == null) {
                    return;
                }
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
                if (mCameraId == CameraCharacteristics.LENS_FACING_FRONT) {
                    rotateMatrix.postRotate(90.0f);
                } else if (mCameraId == CameraCharacteristics.LENS_FACING_BACK) {
                    rotateMatrix.postRotate(270.0f);
                    rotateMatrix.postScale(-1.0f, 1.0f);
                }

                Bitmap rotatedBitmap = Bitmap.createBitmap(
                        imageBitmap, 0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT,
                        rotateMatrix, true
                );
                processImage(rotatedBitmap);
                image.close();
            }
        }
    };


    /**
     * Creates a new [CameraCaptureSession] for camera preview.
     */
    private void createCameraPreviewSession() {
        try {
            // This is the surface we need to record images for processing.
            Surface recordingSurface = imageReader.getSurface();

            // We set up a CaptureRequest.Builder with the output Surface.
            previewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            previewRequestBuilder.addTarget(recordingSurface);

            // Here, we create a CameraCaptureSession for camera preview.
            cameraDevice.createCaptureSession(
                    Arrays.asList(recordingSurface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                            // The camera is already closed
                            if (cameraDevice == null) {
                                return;
                            }

                            // When the session is ready, we start displaying the preview.
                            captureSession = cameraCaptureSession;
                            try {
                                // Auto focus should be continuous for camera preview.
                                previewRequestBuilder.set(
                                        CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                                );
                                // Flash is automatically enabled when necessary.
                                setAutoFlash(previewRequestBuilder);

                                // Finally, we start displaying the camera preview.
                                previewRequest = previewRequestBuilder.build();
                                captureSession.setRepeatingRequest(
                                        previewRequest,
                                        captureCallback, backgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                        }
                    },
                    null);
        } catch (CameraAccessException e) {
            Log.e(TAG, e.toString());
        }
    }

    private void setAutoFlash(CaptureRequest.Builder requestBuilder) {
        if (flashSupported) {
            requestBuilder.set(
                    CaptureRequest.CONTROL_AE_MODE,
                    CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);
        }
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
        paint.setColor(getResources().getColor(R.color.posenet_text_blue));
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

        float widthRatio = (float) screenWidth / MODEL_WIDTH;
        float heightRatio = (float) screenHeight / MODEL_HEIGHT;

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

        canvas.drawText(String.format(getString(R.string.posenet_score) + "%.2f", person.score),
                (15.0f * widthRatio), (30.0f * heightRatio + bottom), paint);
        canvas.drawText(String.format(getString(R.string.posenet_time) + "%.2f ms", posenet.lastInferenceTimeNanos * 1.0f / 1_000_000),
                (15.0f * widthRatio), (50.0f * heightRatio + bottom), paint);

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
        if (null == person) {
            isPreBackgroundThreadPause = true;
            Toast.makeText(getActivity(), R.string.posenet_exit, Toast.LENGTH_LONG).show();
            getActivity().runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    getActivity().finish();
                }
            });

        } else {
            isPreBackgroundThreadPause = false;
            Canvas canvas = surfaceView.getHolder().lockCanvas();
            draw(canvas, person, scaledBitmap);
        }
    }


}