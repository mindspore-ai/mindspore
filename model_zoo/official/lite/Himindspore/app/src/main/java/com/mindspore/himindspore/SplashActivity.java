package com.mindspore.himindspore;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;

import com.mindspore.himindspore.base.BaseActivity;
import com.mindspore.himindspore.imageclassification.ui.ImageMainActivity;
import com.mindspore.himindspore.mvp.MainContract;
import com.mindspore.himindspore.mvp.MainPresenter;
import com.mindspore.himindspore.net.FileDownLoadObserver;
import com.mindspore.himindspore.net.UpdateInfoBean;
import com.mindspore.himindspore.objectdetection.ui.ObjectDetectionMainActivity;

import java.io.File;

public class SplashActivity extends BaseActivity<MainPresenter> implements MainContract.View, View.OnClickListener {

    private static final String TAG = "SplashActivity";
    private static final int REQUEST_PERMISSION = 1;

    private Button btnImage, btnObject, btnContract, btnAdvice;
    private boolean isHasPermssion;

    private ProgressDialog progressDialog;

    private static final String CODE_URL = "https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/lite";
    private static final String HELP_URL = "https://github.com/mindspore-ai/mindspore/issues";


    @Override
    protected void init() {
        presenter = new MainPresenter(this);

        btnImage = findViewById(R.id.btn_image);
        btnObject = findViewById(R.id.btn_object);
        btnContract = findViewById(R.id.btn_contact);
        btnAdvice = findViewById(R.id.btn_advice);

        btnImage.setOnClickListener(this);
        btnObject.setOnClickListener(this);
        btnContract.setOnClickListener(this);
        btnAdvice.setOnClickListener(this);

        requestPermissions();
        getUpdateInfo();
    }

    @Override
    public int getLayout() {
        return R.layout.activity_splash;
    }

    private void requestPermissions() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE,
                        Manifest.permission.READ_PHONE_STATE, Manifest.permission.CAMERA}, REQUEST_PERMISSION);
    }

    /**
     * Authority application result callback
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (REQUEST_PERMISSION == requestCode) {
            isHasPermssion = true;
        }
    }

    private void getUpdateInfo() {
        presenter.getUpdateInfo();
    }


    @Override
    public void onClick(View view) {
        if (R.id.btn_image == view.getId()) {
            if (isHasPermssion) {
                startActivity(new Intent(SplashActivity.this, ImageMainActivity.class));
            } else {
                requestPermissions();
            }
        } else if (R.id.btn_object == view.getId()) {
            if (isHasPermssion) {
                startActivity(new Intent(SplashActivity.this, ObjectDetectionMainActivity.class));
            } else {
                requestPermissions();
            }
        } else if (R.id.btn_contact == view.getId()) {
            openBrowser(CODE_URL);
        } else if (R.id.btn_advice == view.getId()) {
            openBrowser(HELP_URL);
        }
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
        builder.setTitle("下载完成");
        builder.setMessage("是否安装");
        builder.setCancelable(false);
        builder.setPositiveButton("确定", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
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
            }
        });
        builder.setNegativeButton("取消", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
            }
        });
        builder.create().show();
    }


    private int now_version;

    public void showUpdate(final UpdateInfoBean updateInfo) {
        try {
            PackageManager packageManager = this.getPackageManager();
            PackageInfo packageInfo = packageManager.getPackageInfo(this.getPackageName(), 0);
            now_version = packageInfo.versionCode;
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
        }

        if (now_version == updateInfo.getVersionCode()) {
            Toast.makeText(this, "已经是最新版本", Toast.LENGTH_SHORT).show();
            Log.d(TAG + "版本号是", "onResponse: " + now_version);
        } else {
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setIcon(android.R.drawable.ic_dialog_info);
            builder.setTitle("请升级新版本" + updateInfo.getVersionName());
            builder.setMessage(updateInfo.getMessage());
            builder.setCancelable(false);
            builder.setPositiveButton("确定", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    Log.e(TAG, String.valueOf(Environment.MEDIA_MOUNTED));
                    downFile();
                }
            });
            builder.setNegativeButton("取消", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                }
            });
            builder.create().show();
        }
    }

    public void downFile() {
        progressDialog = new ProgressDialog(this);
        progressDialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
        progressDialog.setTitle("正在下载");
        progressDialog.setMessage("请稍候...");
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
                Toast.makeText(SplashActivity.this, "下载失败", Toast.LENGTH_LONG).show();
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
        Log.e("测试路径", directoryPath);
        if (!file.exists()) {
            file.mkdirs();
        }
        return directoryPath;
    }

}