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

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.hardware.camera2.CameraCharacteristics;
import android.media.Image;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.view.SurfaceView;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
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
public class PosenetMainActivity extends AppCompatActivity  {

    private static final String[] PERMISSIONS = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_PHONE_STATE, Manifest.permission.CAMERA};
    private static final int REQUEST_PERMISSION = 1;
    private static final int REQUEST_PERMISSION_AGAIN = 2;
    private boolean isAllGranted;

    private PoseNetFragment poseNetFragment;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.posenet_activity_main);
        addCameraFragment();
    }

    private void addCameraFragment() {
        poseNetFragment = PoseNetFragment.newInstance();
        getSupportFragmentManager().popBackStack();
        getSupportFragmentManager().beginTransaction()
                .replace(R.id.container, poseNetFragment)
                .commitAllowingStateLoss();
    }

    public void onClickSwitch(View view) {
        poseNetFragment.switchCamera();
    }

}