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
package com.mindspore.posenetdemo;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

public class TestActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test);

        ImageView sampleImageView = findViewById(R.id.image);
        Drawable drawedImage = getResources().getDrawable(R.drawable.image);
        Bitmap imageBitmap = drawableToBitmap(drawedImage);
        sampleImageView.setImageBitmap(imageBitmap);
        Posenet posenet = new Posenet(this);
        Posenet.Person person = posenet.estimateSinglePose(imageBitmap);

        // Draw the keypoints over the image.
        Paint paint = new Paint();
        paint.setColor(Color.RED);

        Bitmap mutableBitmap = imageBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        for (Posenet.KeyPoint keypoint : person.keyPoints) {
            canvas.drawCircle(
                    keypoint.position.x,
                    keypoint.position.y, 2.0f, paint);
        }


        sampleImageView.setAdjustViewBounds(true);
        sampleImageView.setImageBitmap(mutableBitmap);
    }


    private Bitmap drawableToBitmap(Drawable drawable) {
        Bitmap bitmap = Bitmap.createBitmap(257, 257, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        drawable.setBounds(0, 0, canvas.getWidth(), canvas.getHeight());
        drawable.draw(canvas);
        return bitmap;
    }
}