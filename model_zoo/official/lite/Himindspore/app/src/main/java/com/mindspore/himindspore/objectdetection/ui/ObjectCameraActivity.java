package com.mindspore.himindspore.objectdetection.ui;

import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.himindspore.R;
import com.mindspore.himindspore.camera.CameraPreview;
import com.mindspore.himindspore.objectdetection.bean.RecognitionObjectBean;
import com.mindspore.himindspore.objectdetection.help.ObjectTrackingMobile;

import java.io.FileNotFoundException;
import java.util.List;

import static com.mindspore.himindspore.objectdetection.bean.RecognitionObjectBean.getRecognitionList;


/**
 * main page of entrance
 * <p>
 * Pass in pictures to JNI, test mindspore model, load reasoning, etc
 */

public class ObjectCameraActivity extends AppCompatActivity implements CameraPreview.RecognitionDataCallBack {

    private final String TAG = "ObjectCameraActivity";

    private CameraPreview cameraPreview;

    private ObjectTrackingMobile mTrackingMobile;

    private ObjectRectView mObjectRectView;

    private List<RecognitionObjectBean> recognitionObjectBeanList;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_object_camera);

        cameraPreview = findViewById(R.id.camera_preview);
        mObjectRectView = findViewById(R.id.objRectView);

        init();
    }

    private void init() {
        try {
            mTrackingMobile = new ObjectTrackingMobile(this);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        boolean ret = mTrackingMobile.loadModelFromBuf(getAssets());
        Log.d(TAG, "TrackingMobile loadModelFromBuf: " + ret);

        cameraPreview.addImageRecognitionDataCallBack(this);
    }


    @Override
    protected void onResume() {
        super.onResume();
        cameraPreview.onResume(this, CameraPreview.OPEN_TYPE_OBJECT, mTrackingMobile);

    }

    @Override
    protected void onPause() {
        super.onPause();
        cameraPreview.onPause();
    }

    @Override
    public void onRecognitionDataCallBack(String result, String time) {
        if (TextUtils.isEmpty(result)) {
            mObjectRectView.clearCanvas();
            return;
        }
        Log.d(TAG, result);
        recognitionObjectBeanList = getRecognitionList(result);
        mObjectRectView.setInfo(recognitionObjectBeanList);
    }
}
