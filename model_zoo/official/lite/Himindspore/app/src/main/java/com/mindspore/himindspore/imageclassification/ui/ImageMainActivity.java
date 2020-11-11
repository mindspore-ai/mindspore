package com.mindspore.himindspore.imageclassification.ui;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.himindspore.R;

public class ImageMainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_main);


        findViewById(R.id.btn_demo).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(ImageMainActivity.this, ImageCameraActivity.class);
                intent.putExtra(ImageCameraActivity.OPEN_TYPE, ImageCameraActivity.TYPE_DEMO);
                startActivity(intent);
            }
        });

        findViewById(R.id.btn_custom).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(ImageMainActivity.this, ImageCameraActivity.class);
                intent.putExtra(ImageCameraActivity.OPEN_TYPE, ImageCameraActivity.TYPE_CUSTOM);
                startActivity(intent);
            }
        });
    }
}