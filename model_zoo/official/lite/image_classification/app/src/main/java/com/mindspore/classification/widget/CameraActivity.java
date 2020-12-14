/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mindspore.classification.widget;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.mindspore.classification.R;
import com.mindspore.classification.gallery.classify.RecognitionObjectBean;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * The main interface of camera preview.
 * Using Camera 2 API.
 */
public class CameraActivity extends AppCompatActivity{
    private static final String TAG = "CameraActivity";

    private static final String[] PERMISSIONS = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_PHONE_STATE, Manifest.permission.CAMERA};
    private static final int REQUEST_PERMISSION = 1;
    private static final int REQUEST_PERMISSION_AGAIN = 2;
    private boolean isAllGranted;

    private static final String BUNDLE_FRAGMENTS_KEY = "android:support:fragments";

    private LinearLayout bottomLayout;

    private List<RecognitionObjectBean> recognitionObjectBeanList;


    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        Log.d(TAG, "onCreate");

        if (savedInstanceState != null && this.clearFragmentsTag()) {
            // Clear the state of the fragment when rebuilding.
            savedInstanceState.remove(BUNDLE_FRAGMENTS_KEY);
        }
        super.onCreate(null);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_camera);
        bottomLayout = findViewById(R.id.layout_bottom_content);
        requestPermissions();
    }

    private void requestPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            isAllGranted = checkPermissionAllGranted(PERMISSIONS);
            if (!isAllGranted) {
                ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_PERMISSION);
            } else {
                addCameraFragment();
            }
        } else {
            isAllGranted = true;
            addCameraFragment();
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
            } else {
                addCameraFragment();
            }
        }
    }

    private void openAppDetails() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage("PoseNet 需要访问 “相机” 和 “外部存储器”，请到 “应用信息 -> 权限” 中授予！");
        builder.setPositiveButton("去手动授权", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Intent intent = new Intent();
                intent.setAction(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                intent.addCategory(Intent.CATEGORY_DEFAULT);
                intent.setData(Uri.parse("package:" + getPackageName()));
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                intent.addFlags(Intent.FLAG_ACTIVITY_NO_HISTORY);
                intent.addFlags(Intent.FLAG_ACTIVITY_EXCLUDE_FROM_RECENTS);
                startActivityForResult(intent, REQUEST_PERMISSION_AGAIN);
            }
        });
        builder.setNegativeButton("取消", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                finish();
            }
        });
        builder.show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (REQUEST_PERMISSION_AGAIN == requestCode) {
            requestPermissions();
        }
    }


    protected void addCameraFragment() {
        CameraFragment cameraFragment = CameraFragment.newInstance(new CameraFragment.RecognitionDataCallBack() {
            @Override
            public void onRecognitionDataCallBack(String result, final String time) {
                dealRecognitionData(result,time);
            }
        });

        getSupportFragmentManager().beginTransaction()
                .replace(R.id.container, cameraFragment)
                .commitAllowingStateLoss();
    }

    private void dealRecognitionData(String result, final String time) {
        if (recognitionObjectBeanList != null) {
            recognitionObjectBeanList.clear();
        } else {
            recognitionObjectBeanList = new ArrayList<>();
        }

        if (!result.equals("")) {
            String[] resultArray = result.split(";");
            for (String singleRecognitionResult:resultArray) {
                String[] singleResult = singleRecognitionResult.split(":");
                float score = Float.parseFloat(singleResult[1]);
                if (score > 0.5) {
                    recognitionObjectBeanList.add(new RecognitionObjectBean(singleResult[0], score));
                }
            }
            Collections.sort(recognitionObjectBeanList, new Comparator<RecognitionObjectBean>() {
                @Override
                public int compare(RecognitionObjectBean t1, RecognitionObjectBean t2) {
                    return Float.compare(t2.getScore(), t1.getScore());
                }
            });
        }

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                showResultsInBottomSheet(recognitionObjectBeanList,time);
            }
        });
    }

    @UiThread
    protected void showResultsInBottomSheet(List<RecognitionObjectBean> list,String time) {
        bottomLayout.removeAllViews();
        if (list != null && list.size() > 0){
            int classNum = 0;
            for (RecognitionObjectBean bean:list) {
                classNum++;
                HorTextView horTextView = new HorTextView(this);
                horTextView.setLeftTitle(bean.getName());
                horTextView.setRightContent(String.format("%.2f", (100 * bean.getScore())) + "%");
                horTextView.setBottomLineVisible(View.VISIBLE);
                bottomLayout.addView(horTextView);
                if (classNum > 4){ // set maximum display is 5.
                    break;
                }
            }
            HorTextView horTextView = new HorTextView(this);
            horTextView.setLeftTitle("Inference Time：");
            horTextView.setRightContent(time);
            horTextView.setBottomLineVisible(View.INVISIBLE);
            bottomLayout.addView(horTextView);
        }else{
            TextView textView = new TextView(this);
            textView.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,ViewGroup.LayoutParams.WRAP_CONTENT));
            textView.setText("Keep moving.");
            textView.setGravity(Gravity.CENTER);
            textView.setTextColor(Color.BLACK);
            textView.setTextSize(30);
            bottomLayout.addView(textView);
        }


    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        if (outState != null && this.clearFragmentsTag()) {
            outState.remove(BUNDLE_FRAGMENTS_KEY);
        }
    }

    protected boolean clearFragmentsTag() {
        return true;
    }
}
