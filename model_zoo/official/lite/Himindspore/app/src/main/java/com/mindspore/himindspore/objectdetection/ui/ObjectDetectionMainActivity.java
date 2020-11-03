package com.mindspore.himindspore.objectdetection.ui;

import android.Manifest;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.mindspore.himindspore.R;

public class ObjectDetectionMainActivity extends AppCompatActivity implements View.OnClickListener {

    private static final int REQUEST_CAMERA_PERMISSION = 2;
    private static final int REQUEST_PHOTO_PERMISSION = 3;

    private Button btnPhoto, btnCamera;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_object_detection_main);

        btnPhoto = findViewById(R.id.btn_photo);
        btnCamera = findViewById(R.id.btn_camera);

        btnPhoto.setOnClickListener(this);
        btnCamera.setOnClickListener(this);
    }


    @Override
    public void onClick(View view) {
        if (R.id.btn_photo == view.getId()) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.READ_PHONE_STATE}, REQUEST_PHOTO_PERMISSION);
        } else if (R.id.btn_camera == view.getId()) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.READ_PHONE_STATE, Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        }
    }

    /**
     * Authority application result callback
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (REQUEST_PHOTO_PERMISSION == requestCode) {
            choosePhoto();
        } else if (REQUEST_CAMERA_PERMISSION == requestCode) {
            chooseCamera();
        }
    }


    private void choosePhoto() {
        Intent intent = new Intent(ObjectDetectionMainActivity.this, ObjectPhotoActivity.class);
        startActivity(intent);
    }

    private void chooseCamera() {
        Intent intent = new Intent(ObjectDetectionMainActivity.this, ObjectCameraActivity.class);
        startActivity(intent);
    }
}

