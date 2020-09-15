package com.mindspore.hiobject.objectdetect;

import android.os.Bundle;
import android.view.WindowManager;

import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.hiobject.R;

/**
 * [入口主页面]
 *
 * 向JNI传入图片，测试MindSpore模型加载推理等.
 */

public class CameraActivity extends AppCompatActivity {

    private final String TAG = "CameraActivity";
    private static final String BUNDLE_FRAGMENTS_KEY = "android:support:fragments";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        if (savedInstanceState != null && this.clearFragmentsTag()) {
            // 重建时清除 fragment的状态
            savedInstanceState.remove(BUNDLE_FRAGMENTS_KEY);
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getSupportFragmentManager().beginTransaction().replace(R.id.container, CameraFragment.newInstance()).commit();
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        if (outState != null && this.clearFragmentsTag()) {
            // 销毁时不保存fragment的状态
            outState.remove(BUNDLE_FRAGMENTS_KEY);
        }
    }

    protected boolean clearFragmentsTag() {
        return true;
    }
}
