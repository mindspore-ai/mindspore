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
import android.graphics.Paint;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.util.Pair;

import java.util.Arrays;
import java.util.List;

import static com.mindspore.posenetdemo.Posenet.BodyPart.LEFT_ANKLE;
import static com.mindspore.posenetdemo.Posenet.BodyPart.LEFT_ELBOW;
import static com.mindspore.posenetdemo.Posenet.BodyPart.LEFT_HIP;
import static com.mindspore.posenetdemo.Posenet.BodyPart.LEFT_KNEE;
import static com.mindspore.posenetdemo.Posenet.BodyPart.LEFT_SHOULDER;
import static com.mindspore.posenetdemo.Posenet.BodyPart.LEFT_WRIST;
import static com.mindspore.posenetdemo.Posenet.BodyPart.RIGHT_ANKLE;
import static com.mindspore.posenetdemo.Posenet.BodyPart.RIGHT_ELBOW;
import static com.mindspore.posenetdemo.Posenet.BodyPart.RIGHT_HIP;
import static com.mindspore.posenetdemo.Posenet.BodyPart.RIGHT_KNEE;
import static com.mindspore.posenetdemo.Posenet.BodyPart.RIGHT_SHOULDER;
import static com.mindspore.posenetdemo.Posenet.BodyPart.RIGHT_WRIST;

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
        paint.setColor(getResources().getColor(R.color.text_blue));
        paint.setTextSize(80.0f);
        paint.setStrokeWidth(5.0f);

        Bitmap mutableBitmap = imageBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);


        for (Posenet.KeyPoint keypoint : person.keyPoints) {
            canvas.drawCircle(
                    keypoint.position.x,
                    keypoint.position.y, 2.0f, paint);
        }
        for (int i = 0; i < bodyJoints.size(); i++) {
            Pair line = (Pair) bodyJoints.get(i);
            Posenet.BodyPart first = (Posenet.BodyPart) line.first;
            Posenet.BodyPart second = (Posenet.BodyPart) line.second;

            if (person.keyPoints.get(first.ordinal()).score > minConfidence &
                    person.keyPoints.get(second.ordinal()).score > minConfidence) {
                canvas.drawLine(
                        person.keyPoints.get(first.ordinal()).position.x,
                        person.keyPoints.get(first.ordinal()).position.y,
                        person.keyPoints.get(second.ordinal()).position.x,
                        person.keyPoints.get(second.ordinal()).position.y, paint);
            }
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

    private final static int MODEL_WIDTH = 257;
    private final static int MODEL_HEIGHT = 257;
    private final double minConfidence = 0.5;
    private final float circleRadius = 8.0f;

    private final List bodyJoints = Arrays.asList(
            new Pair(LEFT_WRIST, LEFT_ELBOW), new Pair(LEFT_ELBOW, LEFT_SHOULDER),
            new Pair(LEFT_SHOULDER, RIGHT_SHOULDER), new Pair(RIGHT_SHOULDER, RIGHT_ELBOW),
            new Pair(RIGHT_ELBOW, RIGHT_WRIST), new Pair(LEFT_SHOULDER, LEFT_HIP),
            new Pair(LEFT_HIP, RIGHT_HIP), new Pair(RIGHT_HIP, RIGHT_SHOULDER),
            new Pair(LEFT_HIP, LEFT_KNEE), new Pair(LEFT_KNEE, LEFT_ANKLE),
            new Pair(RIGHT_HIP, RIGHT_KNEE), new Pair(RIGHT_KNEE, RIGHT_ANKLE));
}