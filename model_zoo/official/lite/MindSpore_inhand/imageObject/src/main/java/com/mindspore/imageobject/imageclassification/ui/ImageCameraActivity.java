/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.UiThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.alibaba.android.arouter.facade.annotation.Autowired;
import com.alibaba.android.arouter.facade.annotation.Route;
import com.alibaba.android.arouter.launcher.ARouter;
import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.common.utils.Utils;
import com.mindspore.customview.dialog.NoticeDialog;
import com.mindspore.imageobject.R;
import com.mindspore.imageobject.camera.CameraPreview;
import com.mindspore.imageobject.imageclassification.bean.RecognitionImageBean;
import com.mindspore.imageobject.imageclassification.help.GarbageTrackingMobile;
import com.mindspore.imageobject.imageclassification.help.ImageTrackingMobile;
import com.mindspore.imageobject.imageclassification.help.SceneTrackingMobile;

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
    private CameraPreview cameraPreview;
    private ImageTrackingMobile imageTrackingMobile;
    private GarbageTrackingMobile garbageTrackingMobile;
    private SceneTrackingMobile sceneTrackingMobile;
    private RecognitionImageBean bean;
    private NoticeDialog noticeDialog;

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
        Toolbar mToolbar = findViewById(R.id.image_camera_toolbar);
        if (TYPE_IMAGE == enterType) {
            mToolbar.setTitle(getString(R.string.image_camera_title));
            imageTrackingMobile = new ImageTrackingMobile(this);
            ret = imageTrackingMobile.loadModelFromBuf(IMAGE_SCENE_MS);
        } else if (TYPE_GARBAGE == enterType) {
            mToolbar.setTitle(getString(R.string.image_garbage_title));
            garbageTrackingMobile = new GarbageTrackingMobile(this);
            ret = garbageTrackingMobile.loadModelFromBuf(GARBAGE_MS);
        } else if (TYPE_SCENE == enterType) {
            mToolbar.setTitle(getString(R.string.image_scene_title));
            sceneTrackingMobile = new SceneTrackingMobile(this);
            ret = sceneTrackingMobile.loadModelFromBuf(IMAGE_SCENE_MS);
        }
        Log.d(TAG, "Loading model return value: " + ret);
        cameraPreview.addImageRecognitionDataCallBack(this);

        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu info");
        getMenuInflater().inflate(R.menu.menu_setting_app, menu);
        if (TYPE_GARBAGE == enterType) {
            menu.removeItem(R.id.item_more);
        }
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int itemId = item.getItemId();
        if (itemId == R.id.item_help) {
            showHelpDialog();
        } else if (itemId == R.id.item_more) {
            if (TYPE_IMAGE == enterType) {
                Utils.openBrowser(this, MSLinkUtils.HELP_IMAGE_CLASSIFICATION);
            } else if (TYPE_SCENE == enterType) {
                Utils.openBrowser(this, MSLinkUtils.HELP_SCENE_DETECTION);
            }
        }
        return super.onOptionsItemSelected(item);
    }

    private void showHelpDialog() {
        noticeDialog = new NoticeDialog(this);
        noticeDialog.setTitleString(getString(R.string.explain_title));
        if (TYPE_IMAGE == enterType) {
            noticeDialog.setContentString(getString(R.string.explain_image_classification));
        } else if (TYPE_GARBAGE == enterType) {
            noticeDialog.setContentString(getString(R.string.explain_garbage_classification));
        } else if (TYPE_SCENE == enterType) {
            noticeDialog.setContentString(getString(R.string.explain_scene_detection));
        }
        noticeDialog.setYesOnclickListener(() -> {
            noticeDialog.dismiss();
        });
        noticeDialog.show();
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
            Log.d(TAG, "imageTrackingMobile Unload model return value: " + ret);
        }
        if (garbageTrackingMobile != null) {
            boolean ret = garbageTrackingMobile.unloadModel();
            Log.d(TAG, "garbageTrackingMobile Unload model return value: " + ret);
        }
        if (sceneTrackingMobile != null) {
            boolean ret = sceneTrackingMobile.unloadModel();
            Log.d(TAG, "sceneTrackingMobile Unload model return value: " + ret);
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
            int maxIndex = Integer.parseInt(result);
            String[] IMAGECONTENT = getResources().getStringArray(R.array.image_category_old);
            runOnUiThread(() -> showResultsInImageGarbage(IMAGECONTENT[maxIndex], time));
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
            runOnUiThread(() -> showResultsInImageGarbage(finalCategoryScore, time));
        } else if (TYPE_SCENE == enterType) {
            if (!result.equals("") && result.contains(":")) {
                String[] resultArray = result.split(":");
                int lableIndex = Integer.parseInt(resultArray[0]);
                float score = Float.parseFloat(resultArray[1]);
                String[] SCENEOBJECT = getResources().getStringArray(R.array.scene_category);
                bean = new RecognitionImageBean(SCENEOBJECT[lableIndex], score);
            }
            runOnUiThread(() -> showResultsInScene(bean, time));
        }
    }

    @UiThread
    protected void showResultsInImageGarbage(String result, String time) {
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
    protected void showResultsInScene(RecognitionImageBean recognitionObjectBean, String time) {
        bottomLayout.removeAllViews();
        if (recognitionObjectBean != null) {
            HorTextView horTextView = new HorTextView(this);
            horTextView.setLeftTitle(recognitionObjectBean.getName() + ":");
            horTextView.setRightContent(String.format("%.2f", (100 * recognitionObjectBean.getScore())) + "%");
            horTextView.getTvRightContent().setTextColor(getResources().getColor(R.color.btn_small_checked));
            horTextView.setBottomLineVisible(View.VISIBLE);
            bottomLayout.addView(horTextView);

            HorTextView horTimeView = new HorTextView(this);
            horTimeView.setLeftTitle(getResources().getString(R.string.title_time));
            horTimeView.setRightContent(time);
            horTimeView.getTvRightContent().setTextColor(getResources().getColor(R.color.btn_small_checked));
            horTimeView.setBottomLineVisible(View.INVISIBLE);
            bottomLayout.addView(horTimeView);
        } else {
            showLoadView();
        }
    }

    private void showLoadView() {
        TextView textView = new TextView(this);
        textView.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        textView.setText(getResources().getString(R.string.title_keep));
        textView.setGravity(Gravity.CENTER);
        textView.setTextColor(Color.BLACK);
        textView.setTextSize(30);
        bottomLayout.addView(textView);
    }

}
