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
package com.mindspore.himindspore.ui.guide;

import android.Manifest;
import android.content.Intent;
import android.net.Uri;
import android.os.Handler;
import android.provider.Settings;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;

import com.mindspore.customview.countdown.CountDownView;
import com.mindspore.himindspore.R;
import com.mindspore.himindspore.base.BaseActivity;
import com.mindspore.himindspore.ui.main.MainActivity;

import java.util.List;

import pub.devrel.easypermissions.AfterPermissionGranted;
import pub.devrel.easypermissions.AppSettingsDialog;
import pub.devrel.easypermissions.EasyPermissions;

public class SplashActivity extends BaseActivity implements EasyPermissions.PermissionCallbacks {

    private static final String TAG = "SplashActivity";

    private static final String[] PERMISSIONS = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_PHONE_STATE, Manifest.permission.CAMERA};
    private static final int REQUEST_PERMISSION = 1;

    private CountDownView cdvTime;

    @Override
    protected void init() {
        cdvTime = findViewById(R.id.cdv_time);
        initCountDownView();
    }

    private void initCountDownView() {
        cdvTime.setTime(3);
        cdvTime.start();
        cdvTime.setOnLoadingFinishListener(() -> startPermissionsTask());
        cdvTime.setOnClickListener(view -> {
            cdvTime.stop();
            startPermissionsTask();
        });
    }

    @Override
    public int getLayout() {
        return R.layout.activity_splash;
    }


    @AfterPermissionGranted(REQUEST_PERMISSION)
    private void startPermissionsTask() {
        if (hasPermissions()) {
            setHandler();
        } else {
            EasyPermissions.requestPermissions(this,
                    this.getResources().getString(R.string.app_need_permission),
                    REQUEST_PERMISSION, PERMISSIONS);
        }
    }

    private boolean hasPermissions() {
        return EasyPermissions.hasPermissions(this, PERMISSIONS);
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        EasyPermissions.onRequestPermissionsResult(requestCode,
                permissions, grantResults, SplashActivity.this);
    }

    @Override
    public void onPermissionsGranted(int requestCode, List<String> perms) {

    }

    @Override
    public void onPermissionsDenied(int requestCode, List<String> perms) {
        if (EasyPermissions.somePermissionPermanentlyDenied(SplashActivity.this, perms)) {
            openAppDetails();
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == AppSettingsDialog.DEFAULT_SETTINGS_REQ_CODE) {
            setHandler();
        }
    }

    private void openAppDetails() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage(getResources().getString(R.string.app_need_permission));
        builder.setPositiveButton(getResources().getString(R.string.app_permission_by_hand), (dialog, which) -> {
            Intent intent = new Intent();
            intent.setAction(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
            intent.addCategory(Intent.CATEGORY_DEFAULT);
            intent.setData(Uri.parse("package:" + getPackageName()));
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            intent.addFlags(Intent.FLAG_ACTIVITY_NO_HISTORY);
            intent.addFlags(Intent.FLAG_ACTIVITY_EXCLUDE_FROM_RECENTS);
            startActivity(intent);
        });
        builder.setNegativeButton(getResources().getString(R.string.cancel), null);
        builder.show();
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK) {
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    private void setHandler() {
        enterMainView();
    }

    private void enterMainView() {
        Intent intent = new Intent();
        intent.setClass(SplashActivity.this, MainActivity.class);
        startActivity(intent);
        finish();
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cdvTime != null && cdvTime.isShown()) {
            cdvTime.stop();
        }
    }
}