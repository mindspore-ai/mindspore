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

package com.mindspore.imageobject.imageclassification.ui;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;

import androidx.annotation.UiThread;
import androidx.appcompat.app.AppCompatActivity;

import com.alibaba.android.arouter.facade.annotation.Autowired;
import com.alibaba.android.arouter.facade.annotation.Route;
import com.alibaba.android.arouter.launcher.ARouter;
import com.mindspore.imageobject.R;
import com.mindspore.imageobject.camera.CameraPreview;
import com.mindspore.imageobject.imageclassification.bean.RecognitionImageBean;
import com.mindspore.imageobject.imageclassification.help.GarbageTrackingMobile;
import com.mindspore.imageobject.imageclassification.help.ImageTrackingMobile;
import com.mindspore.imageobject.imageclassification.help.SceneTrackingMobile;
import com.mindspore.imageobject.util.DisplayUtil;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * The main interface of camera preview.
 * Using Camera 2 API.
 */
@Route(path = "/imageobject/ImageCameraActivity")
public class ImageCameraActivity extends AppCompatActivity implements CameraPreview.RecognitionDataCallBack {
    private static final String TAG = "ImageCameraActivity";
    public static final int TYPE_IMAGE = 1;
    public static final int TYPE_GARBAGE = 2;
    public static final int TYPE_SCENE = 3;

    private static final String IMAGE_SCENE_MS = "model/mobilenetv2.ms";
    private static final String GARBAGE_MS = "model/garbage_mobilenetv2.ms";
    @Autowired(name = "OPEN_TYPE")
    int enterType;

    private LinearLayout bottomLayout;
    private List<RecognitionImageBean> recognitionObjectBeanList;

    private CameraPreview cameraPreview;
    private ImageTrackingMobile imageTrackingMobile;
    private GarbageTrackingMobile garbageTrackingMobile;
    private SceneTrackingMobile sceneTrackingMobile;
    private RecognitionImageBean bean;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //inject
        ARouter.getInstance().inject(this);
        setContentView(R.layout.activity_image_camera);
        cameraPreview = findViewById(R.id.image_camera_preview);
        bottomLayout = findViewById(R.id.layout_bottom_content);
        cameraPreview.setVisibility(View.VISIBLE);
        init();
    }

    private void init() {
        boolean ret = false;
        RelativeLayout.LayoutParams linearParams = (RelativeLayout.LayoutParams) bottomLayout.getLayoutParams();
        if (TYPE_IMAGE == enterType) {
            linearParams.height = DisplayUtil.dip2px(this, 200);
            bottomLayout.setLayoutParams(linearParams);
            imageTrackingMobile = new ImageTrackingMobile(this);
            ret = imageTrackingMobile.loadModelFromBuf(IMAGE_SCENE_MS);
        } else if (TYPE_GARBAGE == enterType) {
            linearParams.height = DisplayUtil.dip2px(this, 100);
            bottomLayout.setLayoutParams(linearParams);
            garbageTrackingMobile = new GarbageTrackingMobile(this);
            ret = garbageTrackingMobile.loadModelFromBuf(GARBAGE_MS);
        } else if (TYPE_SCENE == enterType) {
            linearParams.height = DisplayUtil.dip2px(this, 100);
            bottomLayout.setLayoutParams(linearParams);
            sceneTrackingMobile = new SceneTrackingMobile(this);
            ret = sceneTrackingMobile.loadModelFromBuf(IMAGE_SCENE_MS);
        }
        Log.d(TAG, "Loading model return value: " + ret);
        cameraPreview.addImageRecognitionDataCallBack(this);
    }


    @Override
    protected void onResume() {
        super.onResume();
        cameraPreview.onResume(this);
    }

    @Override
    protected void onPause() {
        super.onPause();
        cameraPreview.onPause();
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (imageTrackingMobile != null) {
            boolean ret = imageTrackingMobile.unloadModel();
            Log.d(TAG, "Unload model return value: " + ret);
        }
        if (garbageTrackingMobile != null) {
            boolean ret = garbageTrackingMobile.unloadModel();
            Log.d(TAG, "garbageTrackingMobile Unload model return value: " + ret);
        }
        if (sceneTrackingMobile != null) {
            boolean ret = sceneTrackingMobile.unloadModel();
            Log.d(TAG, "garbageTrackingMobile Unload model return value: " + ret);
        }
    }

    @Override
    public void onRecognitionBitmapCallBack(Bitmap bitmap) {
        String result = null;
        long startTime = System.currentTimeMillis();
        if (TYPE_IMAGE == enterType) {
            result = imageTrackingMobile.MindSpore_runnet(bitmap);
        } else if (TYPE_GARBAGE == enterType) {
            result = garbageTrackingMobile.MindSpore_runnet(bitmap);
        } else if (TYPE_SCENE == enterType) {
            result = sceneTrackingMobile.MindSpore_runnet(bitmap);
        }
        long endTime = System.currentTimeMillis();

        onRecognitionDataCallBack(result, (endTime - startTime) + "ms ");
        if (!bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }


    public void onRecognitionDataCallBack(final String result, final String time) {
        if (TYPE_IMAGE == enterType) {
            if (recognitionObjectBeanList != null) {
                recognitionObjectBeanList.clear();
            } else {
                recognitionObjectBeanList = new ArrayList<>();
            }

            if (!result.equals("")) {
                String[] resultArray = result.split(";");
                for (String singleRecognitionResult : resultArray) {
                    String[] singleResult = singleRecognitionResult.split(":");
                    int lableIndex = Integer.parseInt(singleResult[0]);
                    float score = Float.parseFloat(singleResult[1]);
                    String[] IMAGEOBJECT = getResources().getStringArray(R.array.image_category);
                    if (score > 0.5) {
                        recognitionObjectBeanList.add(new RecognitionImageBean(IMAGEOBJECT[lableIndex], score));
                    }
                }
                Collections.sort(recognitionObjectBeanList, (t1, t2) -> Float.compare(t2.getScore(), t1.getScore()));
            }
            runOnUiThread(() -> showResultsInBottomSheet(recognitionObjectBeanList, time));
        } else if (TYPE_GARBAGE == enterType) {
            int maxIndex = Integer.parseInt(result);
            String[] GABAGETITLE = getResources().getStringArray(R.array.grbage_sort_map);
            StringBuilder categoryScore = new StringBuilder();
            if (maxIndex <= 9) {
                categoryScore.append(GABAGETITLE[0]);
                categoryScore.append(":");
            } else if (maxIndex > 9 && maxIndex <= 17) {
                categoryScore.append(GABAGETITLE[1]);
                categoryScore.append(":");
            } else if (maxIndex > 17 && maxIndex <= 21) {
                categoryScore.append(GABAGETITLE[2]);
                categoryScore.append(":");
            } else if (maxIndex > 21 && maxIndex <= 25) {
                categoryScore.append(GABAGETITLE[3]);
                categoryScore.append(":");
            }
            categoryScore.append(getResources().getStringArray(R.array.grbage_sort_detailed_map)[maxIndex]);
            String finalCategoryScore = categoryScore.toString();
            runOnUiThread(() -> showResultsInBottomSheetGarbage(finalCategoryScore, time));
        } else if (TYPE_SCENE == enterType) {
            if (!result.equals("") && result.contains(":")) {
                String[] resultArray = result.split(":");
                int lableIndex = Integer.parseInt(resultArray[0]);
                float score = Float.parseFloat(resultArray[1]);
                String[] SCENEOBJECT = getResources().getStringArray(R.array.scene_category);
                bean = new RecognitionImageBean(SCENEOBJECT[lableIndex], score);
            }
            runOnUiThread(() -> showResultsInBottomSheetScene(bean, time));
        }
    }

    @UiThread
    protected void showResultsInBottomSheet(List<RecognitionImageBean> list, String time) {
        bottomLayout.removeAllViews();
        if (list != null && list.size() > 0) {
            int classNum = 0;
            for (RecognitionImageBean bean : list) {
                classNum++;
                HorTextView horTextView = new HorTextView(this);
                horTextView.setLeftTitle(bean.getName());
                horTextView.setRightContent(String.format("%.2f", (100 * bean.getScore())) + "%");
                horTextView.setBottomLineVisible(View.VISIBLE);
                if (classNum == 1) {
                    horTextView.getTvLeftTitle().setTextColor(getResources().getColor(R.color.text_blue));
                    horTextView.getTvRightContent().setTextColor(getResources().getColor(R.color.text_blue));
                } else {
                    horTextView.getTvLeftTitle().setTextColor(getResources().getColor(R.color.white));
                    horTextView.getTvRightContent().setTextColor(getResources().getColor(R.color.white));
                }
                bottomLayout.addView(horTextView);
                if (classNum > 4) { // set maximum display is 5.
                    break;
                }
            }
            HorTextView horTextView = new HorTextView(this);
            horTextView.setLeftTitle(getResources().getString(R.string.title_time));
            horTextView.setRightContent(time);
            horTextView.setBottomLineVisible(View.INVISIBLE);
            horTextView.getTvLeftTitle().setTextColor(getResources().getColor(R.color.text_blue));
            horTextView.getTvRightContent().setTextColor(getResources().getColor(R.color.text_blue));
            bottomLayout.addView(horTextView);
        } else {
            showLoadView();
        }
    }

    @UiThread
    protected void showResultsInBottomSheetGarbage(String result, String time) {
        bottomLayout.removeAllViews();
        if (!TextUtils.isEmpty(result)) {
            HorTextView horTextView = new HorTextView(this);
            horTextView.setLeftTitle(result);
            horTextView.setBottomLineVisible(View.VISIBLE);
            bottomLayout.addView(horTextView);
        } else {
            showLoadView();
        }
    }

    @UiThread
    protected void showResultsInBottomSheetScene(RecognitionImageBean recognitionObjectBean, String time) {
        bottomLayout.removeAllViews();
        if (recognitionObjectBean != null) {
            HorTextView horTextView = new HorTextView(this);
            horTextView.setLeftTitle(recognitionObjectBean.getName() + ":");
            horTextView.setRightContent(String.format("%.2f", (100 * recognitionObjectBean.getScore())) + "%");
            horTextView.setBottomLineVisible(View.VISIBLE);
            bottomLayout.addView(horTextView);

            HorTextView horTimeView = new HorTextView(this);
            horTimeView.setLeftTitle(getResources().getString(R.string.title_time));
            horTimeView.setRightContent(time);
            horTimeView.setBottomLineVisible(View.INVISIBLE);
            bottomLayout.addView(horTimeView);
        } else {
            TextView textView = new TextView(this);
            textView.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
            textView.setText(getResources().getString(R.string.title_keep));
            textView.setGravity(Gravity.CENTER);
            textView.setTextColor(Color.BLACK);
            textView.setTextSize(30);
            bottomLayout.addView(textView);
        }
    }

    private void showLoadView() {
        TextView textView = new TextView(this);
        textView.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        textView.setText(getResources().getString(R.string.title_keep));
        textView.setGravity(Gravity.CENTER);
        textView.setTextColor(Color.WHITE);
        textView.setTextSize(30);
        bottomLayout.addView(textView);
    }
}
