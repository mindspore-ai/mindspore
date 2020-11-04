package com.mindspore.hiobject;

import android.Manifest;
import android.content.Intent;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.mindspore.hiobject.objectdetect.CameraActivity;
import com.mindspore.hiobject.objectdetect.PhotoActivity;

public class SplashActivity extends AppCompatActivity implements View.OnClickListener {

    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int REQUEST_CAMERA_PERMISSION = 2;

    private Button btnPhoto, btnCamera;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

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
                            Manifest.permission.READ_PHONE_STATE}, RC_CHOOSE_PHOTO);
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
        if (RC_CHOOSE_PHOTO == requestCode) {
            choosePhoto();
        } else if (REQUEST_CAMERA_PERMISSION == requestCode) {
            chooseCamera();
        }
    }


    private void choosePhoto() {
        Intent intentToPickPic = new Intent(Intent.ACTION_PICK, null);
        intentToPickPic.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intentToPickPic, RC_CHOOSE_PHOTO);
    }

    private void chooseCamera() {
        Intent intent = new Intent(SplashActivity.this, CameraActivity.class);
        startActivity(intent);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (RC_CHOOSE_PHOTO == requestCode && null != data && null != data.getData()) {
            Intent intent = new Intent(SplashActivity.this, PhotoActivity.class);
            intent.setData(data.getData());
            startActivity(intent);
        }
    }
}

