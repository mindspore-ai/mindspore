package com.mindspore.hiobject.objectdetect;

import android.os.Bundle;
import android.view.WindowManager;

import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.hiobject.R;

/**
 * Main page of entrance
 *
 * Pass in pictures to JNI, test mindspore model, load reasoning, etc
 */

public class CameraActivity extends AppCompatActivity {

    private final String TAG = "CameraActivity";
    private static final String BUNDLE_FRAGMENTS_KEY = "android:support:fragments";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        if (savedInstanceState != null && this.clearFragmentsTag()) {
            savedInstanceState.remove(BUNDLE_FRAGMENTS_KEY);
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getSupportFragmentManager().beginTransaction().replace(R.id.container, CameraFragment.newInstance()).commit();
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
