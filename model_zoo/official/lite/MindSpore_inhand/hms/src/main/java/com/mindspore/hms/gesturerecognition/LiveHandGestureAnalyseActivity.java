/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
package com.mindspore.hms.gesturerecognition;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.util.SparseArray;
import android.view.View;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.huawei.hms.mlsdk.common.LensEngine;
import com.huawei.hms.mlsdk.common.MLAnalyzer;
import com.huawei.hms.mlsdk.gesture.MLGesture;
import com.huawei.hms.mlsdk.gesture.MLGestureAnalyzer;
import com.huawei.hms.mlsdk.gesture.MLGestureAnalyzerFactory;
import com.huawei.hms.mlsdk.gesture.MLGestureAnalyzerSetting;
import com.mindspore.hms.R;
import com.mindspore.hms.camera.GraphicOverlay;
import com.mindspore.hms.camera.HandGestureGraphic;
import com.mindspore.hms.camera.LensEnginePreview;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@Route(path = "/hms/LiveHandGestureAnalyseActivity")
public class LiveHandGestureAnalyseActivity extends AppCompatActivity implements View.OnClickListener {

    private static final String TAG = LiveHandGestureAnalyseActivity.class.getSimpleName();

    private GraphicOverlay mGraphicOverlay;
    private LensEnginePreview mPreview;

    private Button mFacingSwitch;

    private MLGestureAnalyzer mAnalyzer;

    private LensEngine mLensEngine;

    private final int lensType = LensEngine.BACK_LENS;

    private int mLensType;

    private boolean isFront = false;

    private boolean isPermissionRequested;

    private static final int CAMERA_PERMISSION_CODE = 0;

    private static final String[] ALL_PERMISSION =
            new String[]{
                    Manifest.permission.CAMERA,
            };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live_hand_gesture_analyse);
        if (savedInstanceState != null) {
            mLensType = savedInstanceState.getInt("lensType");
        }
        init();
    }

    private void init() {
        Toolbar mToolbar = findViewById(R.id.gesture_activity_toolbar);
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
        mPreview = this.findViewById(R.id.gesture_preview);
        mGraphicOverlay = this.findViewById(R.id.gesture_overlay);
        mFacingSwitch = this.findViewById(R.id.gesture_facingSwitch);
        mFacingSwitch.setOnClickListener(this);
        createHandAnalyzer();
        if (Camera.getNumberOfCameras() == 1) {
            mFacingSwitch.setVisibility(View.GONE);
        }
        // Checking Camera Permissions
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            createLensEngine();
        } else {
            checkPermission();
        }

    }

    private void createHandAnalyzer() {
        // Create a  analyzer. You can create an analyzer using the provided customized face detection parameter: MLHandKeypointAnalyzerSetting
        MLGestureAnalyzerSetting setting =
                new MLGestureAnalyzerSetting.Factory()
                        .create();
        mAnalyzer = MLGestureAnalyzerFactory.getInstance().getGestureAnalyzer(setting);
        mAnalyzer.setTransactor(new HandAnalyzerTransactor(this, mGraphicOverlay));
    }

    // Check the permissions required by the SDK.
    private void checkPermission() {
        if (Build.VERSION.SDK_INT >= 23 && !isPermissionRequested) {
            isPermissionRequested = true;
            ArrayList<String> permissionsList = new ArrayList<>();
            for (String perm : getAllPermission()) {
                if (PackageManager.PERMISSION_GRANTED != this.checkSelfPermission(perm)) {
                    permissionsList.add(perm);
                }
            }

            if (!permissionsList.isEmpty()) {
                requestPermissions(permissionsList.toArray(new String[0]), 0);
            }
        }
    }

    public static List<String> getAllPermission() {
        return Collections.unmodifiableList(Arrays.asList(ALL_PERMISSION));
    }

    private void createLensEngine() {
        Context context = this.getApplicationContext();
        // Create LensEngine.
        mLensEngine = new LensEngine.Creator(context, mAnalyzer)
                .setLensType(this.mLensType)
                .applyDisplayDimension(640, 480)
                .applyFps(25.0f)
                .enableAutomaticFocus(true)
                .create();
    }

    private void startLensEngine() {
        if (this.mLensEngine != null) {
            try {
                this.mPreview.start(this.mLensEngine, this.mGraphicOverlay);
            } catch (IOException e) {
                Log.e(TAG, "Failed to start lens engine.", e);
                this.mLensEngine.release();
                this.mLensEngine = null;
            }
        }
    }

    // Permission application callback.
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        boolean hasAllGranted = true;
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                this.createLensEngine();
            } else if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                hasAllGranted = false;
                if (!ActivityCompat.shouldShowRequestPermissionRationale(this, permissions[0])) {
                    showWaringDialog();
                } else {
                    finish();
                }
            }
            return;
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState) {
        outState.putInt("lensType", this.lensType);
        super.onSaveInstanceState(outState);
    }

    private static class HandAnalyzerTransactor implements MLAnalyzer.MLTransactor<MLGesture> {
        private final GraphicOverlay mGraphicOverlay;

        HandAnalyzerTransactor(LiveHandGestureAnalyseActivity mainActivity, GraphicOverlay ocrGraphicOverlay) {
            this.mGraphicOverlay = ocrGraphicOverlay;
        }

        /**
         * Process the results returned by the analyzer.
         */
        @Override
        public void transactResult(MLAnalyzer.Result<MLGesture> result) {
            this.mGraphicOverlay.clear();

            SparseArray<MLGesture> handGestureSparseArray = result.getAnalyseList();
            List<MLGesture> list = new ArrayList<>();
            for (int i = 0; i < handGestureSparseArray.size(); i++) {
                list.add(handGestureSparseArray.valueAt(i));
            }
            HandGestureGraphic graphic = new HandGestureGraphic(this.mGraphicOverlay, list);
            this.mGraphicOverlay.add(graphic);
        }

        @Override
        public void destroy() {
            this.mGraphicOverlay.clear();
        }

    }

    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.gesture_facingSwitch) {
            switchCamera();
        }
    }

    private void switchCamera() {
        isFront = !isFront;
        if (this.isFront) {
            mLensType = LensEngine.FRONT_LENS;
        } else {
            mLensType = LensEngine.BACK_LENS;
        }
        if (this.mLensEngine != null) {
            this.mLensEngine.close();
        }
        this.createLensEngine();
        this.startLensEngine();
    }

    private void showWaringDialog() {
        AlertDialog.Builder dialog = new AlertDialog.Builder(this);
        dialog.setMessage(R.string.app_need_permission)
                .setPositiveButton(R.string.app_permission_by_hand, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                        Uri uri = Uri.fromParts("package", getApplicationContext().getPackageName(), null);
                        intent.setData(uri);
                        startActivity(intent);
                    }
                })
                .setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        finish();
                    }
                }).setOnCancelListener(dialogInterface);
        dialog.setCancelable(false);
        dialog.show();
    }

    static DialogInterface.OnCancelListener dialogInterface = new DialogInterface.OnCancelListener() {
        @Override
        public void onCancel(DialogInterface dialog) {
        }
    };

    @Override
    protected void onResume() {
        super.onResume();
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            createLensEngine();
            startLensEngine();
        } else {
            checkPermission();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        mPreview.stop();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (this.mLensEngine != null) {
            this.mLensEngine.release();
        }
        if (this.mAnalyzer != null) {
            this.mAnalyzer.stop();
        }
    }
}