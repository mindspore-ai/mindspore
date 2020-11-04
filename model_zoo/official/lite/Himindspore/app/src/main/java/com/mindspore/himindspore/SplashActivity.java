package com.mindspore.himindspore;

import android.Manifest;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.mindspore.himindspore.imageclassification.ui.ImageCameraActivity;
import com.mindspore.himindspore.imageclassification.ui.ImageMainActivity;
import com.mindspore.himindspore.objectdetection.ui.ObjectDetectionMainActivity;

public class SplashActivity extends AppCompatActivity implements View.OnClickListener {

    private static final int REQUEST_PERMISSION = 1;

    private Button btnImage, btnObject, btnContract,btnAdvice;
    private boolean isHasPermssion;

    private static final String CODE_URL ="https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/lite";
    private static final String HELP_URL ="https://github.com/mindspore-ai/mindspore/issues";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        btnImage = findViewById(R.id.btn_image);
        btnObject = findViewById(R.id.btn_object);
        btnContract = findViewById(R.id.btn_contact);
        btnAdvice = findViewById(R.id.btn_advice);

        btnImage.setOnClickListener(this);
        btnObject.setOnClickListener(this);
        btnContract.setOnClickListener(this);
        btnAdvice.setOnClickListener(this);

        requestPermissions();
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
        }else if (R.id.btn_advice == view.getId()) {
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
}