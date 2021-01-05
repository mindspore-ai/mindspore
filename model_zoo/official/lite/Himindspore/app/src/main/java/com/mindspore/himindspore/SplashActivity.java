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
package com.mindspore.himindspore;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.alibaba.android.arouter.launcher.ARouter;
import com.mindspore.himindspore.base.BaseActivity;
import com.mindspore.himindspore.mvp.MainContract;
import com.mindspore.himindspore.mvp.MainPresenter;
import com.mindspore.himindspore.net.FileDownLoadObserver;
import com.mindspore.himindspore.net.UpdateInfoBean;

import java.io.File;

@Route(path = "/himindspore/SplashActivity")
public class SplashActivity extends BaseActivity<MainPresenter> implements MainContract.View {

    private static final String TAG = "SplashActivity";

    private static final String[] PERMISSIONS = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_PHONE_STATE, Manifest.permission.CAMERA};
    private static final int REQUEST_PERMISSION = 1;

    private boolean isAllGranted;
    private int now_version;

    private ProgressDialog progressDialog;
    private TextView versionText;

    private static final String CODE_URL = "https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/lite";
    private static final String HELP_URL = "https://gitee.com/mindspore/mindspore/issues/new?issue%5Bassignee_id%5D=0&issue%5Bmilestone_id%5D=0";
    private static final String STAR_URL = "https://gitee.com/mindspore/mindspore";
    private static final String APK_URL = "https://download.mindspore.cn/model_zoo/official/lite/apk/mindmain.html";


    @Override
    protected void init() {
        presenter = new MainPresenter(this);
        versionText = findViewById(R.id.tv_vision);
        showPackaeInfo();
        requestPermissions();
        getUpdateInfo();
    }

    @Override
    public int getLayout() {
        return R.layout.activity_splash;
    }

    private void showPackaeInfo() {
        try {
            PackageManager packageManager = this.getPackageManager();
            PackageInfo packageInfo = packageManager.getPackageInfo(this.getPackageName(), 0);
            now_version = packageInfo.versionCode;
            versionText.setText(getString(R.string.title_version) + packageInfo.versionName);
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
        }
    }

    private void requestPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            isAllGranted = checkPermissionAllGranted(PERMISSIONS);
            if (!isAllGranted) {
                ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_PERMISSION);
            }
        } else {
            isAllGranted = true;
        }
    }


    private boolean checkPermissionAllGranted(String[] permissions) {
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    /**
     * Authority application result callback
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (REQUEST_PERMISSION == requestCode) {
            isAllGranted = true;

            for (int grant : grantResults) {
                if (grant != PackageManager.PERMISSION_GRANTED) {
                    isAllGranted = false;
                    break;
                }
            }
            if (!isAllGranted) {
                openAppDetails();
            }
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

    private void getUpdateInfo() {
        presenter.getUpdateInfo();
    }


    public void onClickImage(View view) {
        if (isAllGranted) {
            ARouter.getInstance().build("/imageobject/ImageCameraActivity")
                    .withInt("OPEN_TYPE", 1).navigation();
        } else {
            requestPermissions();
        }
    }

    public void onClickGarbage(View view) {
        if (isAllGranted) {
            ARouter.getInstance().build("/imageobject/ImageCameraActivity")
                    .withInt("OPEN_TYPE", 2).navigation();
        } else {
            requestPermissions();
        }
    }

    public void onClickSceneDetection(View view) {
        if (isAllGranted) {
            ARouter.getInstance().build("/imageobject/ImageCameraActivity")
                    .withInt("OPEN_TYPE", 3).navigation();
        } else {
            requestPermissions();
        }
    }


    public void onClickPhotoDetection(View view) {
        if (isAllGranted) {
            ARouter.getInstance().build("/imageobject/ObjectPhotoActivity").navigation();
        } else {
            requestPermissions();
        }
    }

    public void onClickCameraDetection(View view) {
        if (isAllGranted) {
            ARouter.getInstance().build("/imageobject/ObjectCameraActivity").navigation();
        } else {
            requestPermissions();
        }
    }

    public void onClickPoseNet(View view) {
        if (isAllGranted) {
            ARouter.getInstance().build("/posenet/PosenetMainActivity").navigation(this);
        } else {
            requestPermissions();
        }
    }

    public void onClickStyleTransfer(View view) {
        if (isAllGranted) {
            ARouter.getInstance().build("/styletransfer/StyleMainActivity").navigation(this);
        } else {
            requestPermissions();
        }
    }

    public void onClickSegmentation(View view) {
        if (isAllGranted) {
            ARouter.getInstance().build("/segmentation/SegmentationMainActivity").navigation(this);
        } else {
            requestPermissions();
        }
    }

    public void onClickSouceCode(View view) {
        openBrowser(CODE_URL);
    }

    public void onClickHelp(View view) {
        openBrowser(HELP_URL);
    }


    public void onClickStar(View view) {
        openBrowser(STAR_URL);
    }

    public void onClickShare(View view) {
        Intent share_intent = new Intent();
        share_intent.setAction(Intent.ACTION_SEND);
        share_intent.setType("text/plain");
        share_intent.putExtra(Intent.EXTRA_SUBJECT, getString(R.string.title_share));
        share_intent.putExtra(Intent.EXTRA_TEXT, getString(R.string.title_share_commend) + APK_URL);
        share_intent = Intent.createChooser(share_intent, getString(R.string.title_share));
        startActivity(share_intent);
    }

    public void openBrowser(String url) {
        Intent intent = new Intent();
        intent.setAction("android.intent.action.VIEW");
        Uri uri = Uri.parse(url.trim());
        intent.setData(uri);
        startActivity(intent);
    }

    @Override
    public void showUpdateResult(UpdateInfoBean bean) {
        showUpdate(bean);
    }

    @Override
    public void showFail(String s) {

    }

    public void downSuccess() {
        if (progressDialog != null && progressDialog.isShowing()) {
            progressDialog.dismiss();
        }
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setIcon(android.R.drawable.ic_dialog_info);
        builder.setTitle(getResources().getString(R.string.app_download_success));
        builder.setMessage(getResources().getString(R.string.app_need_install));
        builder.setCancelable(false);
        builder.setPositiveButton(getResources().getString(R.string.confirm), (dialog, which) -> {
            Intent intent = new Intent(Intent.ACTION_VIEW);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                intent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                Uri contentUri = FileProvider.getUriForFile(SplashActivity.this, "com.mindspore.himindspore.fileprovider",
                        new File(getApkPath(), "HiMindSpore.apk"));
                intent.setDataAndType(contentUri, "application/vnd.android.package-archive");
            } else {
                intent.setDataAndType(Uri.fromFile(new File(getApkPath(), "HiMindSpore.apk")), "application/vnd.android.package-archive");
                intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            }
            startActivity(intent);
        });
        builder.setNegativeButton(getResources().getString(R.string.cancel), (dialog, which) -> {
        });
        builder.create().show();
    }


    public void showUpdate(final UpdateInfoBean updateInfo) {
        if (now_version != updateInfo.getVersionCode()) {
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setIcon(android.R.drawable.ic_dialog_info);
            builder.setTitle(getResources().getString(R.string.app_update_lastest) + updateInfo.getVersionName());
            builder.setMessage(updateInfo.getMessage());
            builder.setCancelable(false);
            builder.setPositiveButton(getResources().getString(R.string.confirm), (dialog, which) -> {
                Log.e(TAG, String.valueOf(Environment.MEDIA_MOUNTED));
                downFile();
            });
            builder.setNegativeButton(getResources().getString(R.string.cancel), (dialog, which) -> {
            });
            builder.create().show();
        }
    }

    public void downFile() {
        progressDialog = new ProgressDialog(this);
        progressDialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
        progressDialog.setTitle(getResources().getString(R.string.app_is_loading));
        progressDialog.setMessage(getResources().getString(R.string.app_wait));
        progressDialog.setProgressNumberFormat("%1d Mb/%2d Mb");
        progressDialog.setProgress(0);
        progressDialog.show();
        presenter.downloadApk(getApkPath(), "HiMindSpore.apk", new FileDownLoadObserver<File>() {
            @Override
            public void onDownLoadSuccess(File file) {
                downSuccess();
            }

            @Override
            public void onDownLoadFail(Throwable throwable) {
                Toast.makeText(SplashActivity.this, getResources().getString(R.string.app_load_fail), Toast.LENGTH_LONG).show();
            }

            @Override
            public void onProgress(final int progress, final long total) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        progressDialog.setMax((int) total / 1024 / 1024);
                        progressDialog.setProgress(progress);
                    }
                });

            }
        });
        Log.d(TAG, "downFile: ");
    }

    public String getApkPath() {
        String directoryPath = "";
        if (Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState())) {
            directoryPath = getExternalFilesDir("apk").getAbsolutePath();
        } else {
            directoryPath = getFilesDir() + File.separator + "apk";
        }
        File file = new File(directoryPath);
        if (!file.exists()) {
            file.mkdirs();
        }
        return directoryPath;
    }


}