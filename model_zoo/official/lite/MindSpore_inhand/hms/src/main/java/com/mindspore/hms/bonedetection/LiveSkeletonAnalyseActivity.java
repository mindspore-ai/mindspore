package com.mindspore.hms.bonedetection;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.provider.Settings;
import android.util.Log;
import android.util.SparseArray;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.huawei.hms.mlsdk.common.LensEngine;
import com.huawei.hms.mlsdk.common.MLAnalyzer;
import com.huawei.hms.mlsdk.skeleton.MLJoint;
import com.huawei.hms.mlsdk.skeleton.MLSkeleton;
import com.huawei.hms.mlsdk.skeleton.MLSkeletonAnalyzer;
import com.huawei.hms.mlsdk.skeleton.MLSkeletonAnalyzerFactory;
import com.huawei.hms.mlsdk.skeleton.MLSkeletonAnalyzerSetting;
import com.mindspore.hms.R;
import com.mindspore.hms.camera.GraphicOverlay;
import com.mindspore.hms.camera.LensEnginePreview;
import com.mindspore.hms.camera.SkeletonGraphic;
import com.mindspore.hms.camera.SkeletonUtils;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
@Route(path = "/posenet/LiveSkeletonAnalyseActivity")
public class LiveSkeletonAnalyseActivity extends AppCompatActivity implements View.OnClickListener {

    private static final String TAG = LiveSkeletonAnalyseActivity.class.getSimpleName();

    public static final int UPDATE_VIEW = 101;

    private boolean isYogaModeChecked = true;

    private static final int CAMERA_PERMISSION_CODE = 0;

    private final Handler mHandler = new MsgHandler(this);

    private MLSkeletonAnalyzer analyzer;

    private LensEngine mLensEngine;

    private LensEnginePreview mPreview;

    private GraphicOverlay graphicOverlay;

    private ImageView templateImgView;

    private TextView similarityTxt;

    private int lensType = LensEngine.BACK_LENS;

    private boolean isFront = false;

    private List<MLSkeleton> templateList;

    private boolean isPermissionRequested;

    private static final String[] ALL_PERMISSION =
            new String[]{
                    Manifest.permission.CAMERA,
            };

    // Coordinates for the bones of the image template.
    static final float[][] TMP_SKELETONS = {{416.6629f, 312.46442f, 101, 0.8042025f}, {382.3348f, 519.43396f, 102, 0.86383355f},
            {381.0387f, 692.09515f, 103, 0.7551306f}, {659.49194f, 312.24445f, 104, 0.8305682f}, {693.5356f, 519.4844f, 105, 0.8932837f},
            {694.0054f, 692.4169f, 106, 0.8742422f}, {485.08786f, 726.8787f, 107, 0.6004682f}, {485.02808f, 935.4897f, 108, 0.7334503f},
            {485.09384f, 1177.127f, 109, 0.67240065f}, {623.7807f, 726.7474f, 110, 0.5483011f}, {624.5828f, 936.3222f, 111, 0.730425f},
            {625.81915f, 1212.2491f, 112, 0.72417295f}, {521.47363f, 103.95903f, 113, 0.7780853f}, {521.6231f, 277.2533f, 114, 0.7745689f}};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live_skeleton_analyse);
        if (savedInstanceState != null) {
            this.lensType = savedInstanceState.getInt("lensType");
        }
        init();
    }

    private void init() {
        Toolbar mToolbar = findViewById(R.id.posenet_activity_toolbar);
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
        this.mPreview = this.findViewById(R.id.skeleton_preview);
        this.graphicOverlay = this.findViewById(R.id.skeleton_overlay);
        this.createSkeletonAnalyzer(isYogaModeChecked);
        Button facingSwitchBtn = this.findViewById(R.id.skeleton_facingSwitch);
        templateImgView = this.findViewById(R.id.template_imgView);
        templateImgView.setImageResource(R.drawable.skeleton_template);
        similarityTxt = this.findViewById(R.id.similarity_txt);
        if (Camera.getNumberOfCameras() == 1) {
            facingSwitchBtn.setVisibility(View.GONE);
        }
        facingSwitchBtn.setOnClickListener(this);
        initTemplateData();
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            this.createLensEngine();
        } else {
            this.checkPermission();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            this.createLensEngine();
            this.startLensEngine();
        } else {
            this.checkPermission();
        }
    }

    private void createSkeletonAnalyzer(boolean isYogaModeChecked) {
        if (isYogaModeChecked) {
            MLSkeletonAnalyzerSetting setting = new MLSkeletonAnalyzerSetting.Factory()
                    // Set analyzer mode.
                    // MLSkeletonAnalyzerSetting.TYPE_NORMAL:Detect skeleton corresponding to common human posture.
                    // MLSkeletonAnalyzerSetting.TYPE_YOGA：Detect skeleton points corresponding to yoga posture.
                    .setAnalyzerType(MLSkeletonAnalyzerSetting.TYPE_YOGA)
                    .create();
            this.analyzer = MLSkeletonAnalyzerFactory.getInstance().getSkeletonAnalyzer(setting);
        } else {
            this.analyzer = MLSkeletonAnalyzerFactory.getInstance().getSkeletonAnalyzer();
        }
        this.analyzer.setTransactor(new SkeletonAnalyzerTransactor(this, this.graphicOverlay));
    }

    private void createLensEngine() {
        Context context = this.getApplicationContext();
        // Create LensEngine.
        this.mLensEngine = new LensEngine.Creator(context, this.analyzer)
                .setLensType(this.lensType)
                .applyDisplayDimension(1080, 720)
                .applyFps(25.0f)
                .enableAutomaticFocus(true)
                .create();
    }

    private void startLensEngine() {
        if (this.mLensEngine != null) {
            try {
                this.mPreview.start(this.mLensEngine, this.graphicOverlay);
            } catch (IOException e) {
                Log.e(TAG, "Failed to start lens engine.", e);
                this.mLensEngine.release();
                this.mLensEngine = null;
            }
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        this.mPreview.stop();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (this.mLensEngine != null) {
            this.mLensEngine.release();
        }
        if (this.analyzer != null) {
            try {
                this.analyzer.stop();
            } catch (IOException e) {
                Log.e(TAG, "Stop failed: " + e.getMessage());
            }
        }
    }

    // Permission application callback.
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        boolean hasAllGranted = true;
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                this.createLensEngine();
            } else if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                hasAllGranted = false;
                if (!ActivityCompat.shouldShowRequestPermissionRationale(this, permissions[0])) {
                    showWaringDialog();
                } else {
                    finish();
                }
            }
            return;
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    public void onSaveInstanceState(Bundle savedInstanceState) {
        savedInstanceState.putInt("lensType", this.lensType);
        super.onSaveInstanceState(savedInstanceState);
    }

    @Override
    public void onClick(View v) {
        this.isFront = !this.isFront;
        if (this.isFront) {
            this.lensType = LensEngine.FRONT_LENS;
        } else {
            this.lensType = LensEngine.BACK_LENS;
        }
        if (this.mLensEngine != null) {
            this.mLensEngine.close();
        }
        this.createLensEngine();
        this.startLensEngine();
    }

    private void initSwitchListener(Switch analyzerMode) {
        analyzerMode.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton button, boolean isChecked) {
                if (isChecked) {
                    isYogaModeChecked = true;
                    Log.i(TAG, "yoga mode open, isAnalyzerModeChecked = " + isYogaModeChecked);
                } else {
                    isYogaModeChecked = false;
                    Log.i(TAG, "yoga mode close, isAnalyzerModeChecked = " + isYogaModeChecked);
                }
                reStartAnalyzer();
            }
        });
    }

    /**
     * After modifying the skeleton analyzer configuration, you need to create a skeleton analyzer again.
     */
    private void reStartAnalyzer() {
        if (mPreview != null) {
            mPreview.stop();
        }
        if (mLensEngine != null) {
            mLensEngine.release();
        }
        if (analyzer != null) {
            try {
                analyzer.stop();
            } catch (IOException e) {
                Log.e(TAG, e.getMessage());
            }
        }
        Log.i(TAG, "skeleton analyzer recreate, isYogaModeChecked = " + isYogaModeChecked);
        createSkeletonAnalyzer(isYogaModeChecked);
        createLensEngine();
        startLensEngine();
    }

    private void initTemplateData() {
        if (templateList != null) {
            return;
        }
        List<MLJoint> mlJointList = new ArrayList<>();
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.skeleton_template);
        for (int i = 0; i < TMP_SKELETONS.length; i++) {
            MLJoint mlJoint = new MLJoint(bitmap.getWidth() * TMP_SKELETONS[i][0],
                    bitmap.getHeight() * TMP_SKELETONS[i][1], (int) TMP_SKELETONS[i][2], TMP_SKELETONS[i][3]);
            mlJointList.add(mlJoint);
        }

        templateList = new ArrayList<>();
        templateList.add(new MLSkeleton(mlJointList));
    }

    /**
     * Compute Similarity.
     *
     * @param skeletons skeletons
     */
    private void compareSimilarity(List<MLSkeleton> skeletons) {
        if (templateList == null) {
            return;
        }

        float similarity = 0f;
        float result = analyzer.caluteSimilarity(skeletons, templateList);
        if (result > similarity) {
            similarity = result;
        }

        Message msg = Message.obtain();
        Bundle bundle = new Bundle();
        bundle.putFloat("similarity", similarity);
        msg.setData(bundle);
        msg.what = UPDATE_VIEW;
        mHandler.sendMessage(msg);
    }


    private static class MsgHandler extends Handler {
        WeakReference<LiveSkeletonAnalyseActivity> mMainActivityWeakReference;

        MsgHandler(LiveSkeletonAnalyseActivity mainActivity) {
            mMainActivityWeakReference = new WeakReference<>(mainActivity);
        }

        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            LiveSkeletonAnalyseActivity mainActivity = mMainActivityWeakReference.get();
            if (mainActivity == null || mainActivity.isFinishing()) {
                return;
            }
            if (msg.what == UPDATE_VIEW) {
                Bundle bundle = msg.getData();
                float result = bundle.getFloat("similarity");
                mainActivity.similarityTxt.setVisibility(View.VISIBLE);
                mainActivity.similarityTxt.setText("similarity:" + (int) (result * 100) + "%");
            }
        }
    }

    private static class SkeletonAnalyzerTransactor implements MLAnalyzer.MLTransactor<MLSkeleton> {
        private final GraphicOverlay mGraphicOverlay;

        WeakReference<LiveSkeletonAnalyseActivity> mMainActivityWeakReference;

        SkeletonAnalyzerTransactor(LiveSkeletonAnalyseActivity mainActivity, GraphicOverlay ocrGraphicOverlay) {
            mMainActivityWeakReference = new WeakReference<>(mainActivity);
            this.mGraphicOverlay = ocrGraphicOverlay;
        }

        /**
         * Process the results returned by the analyzer.
         */
        @Override
        public void transactResult(MLAnalyzer.Result<MLSkeleton> result) {
            this.mGraphicOverlay.clear();

            SparseArray<MLSkeleton> sparseArray = result.getAnalyseList();
            List<MLSkeleton> list = new ArrayList<>();
            for (int i = 0; i < sparseArray.size(); i++) {
                list.add(sparseArray.valueAt(i));
            }
            // Remove invalid point.
            List<MLSkeleton> skeletons = SkeletonUtils.getValidSkeletons(list);
            SkeletonGraphic graphic = new SkeletonGraphic(this.mGraphicOverlay, skeletons);
            this.mGraphicOverlay.add(graphic);

            LiveSkeletonAnalyseActivity mainActivity = mMainActivityWeakReference.get();
            if (mainActivity != null && !mainActivity.isFinishing()) {
                mainActivity.compareSimilarity(skeletons);
            }
        }

        @Override
        public void destroy() {
            this.mGraphicOverlay.clear();
        }

    }

    // Check the permissions required by the SDK.
    private void checkPermission() {
        if (Build.VERSION.SDK_INT >= 23 && !isPermissionRequested) {
            isPermissionRequested = true;
            ArrayList<String> permissionsList = new ArrayList<>();
            for (String perm : getAllPermission()) {
                if (PackageManager.PERMISSION_GRANTED != this.checkSelfPermission(perm)) {
                    permissionsList.add(perm);
                }
            }

            if (!permissionsList.isEmpty()) {
                requestPermissions(permissionsList.toArray(new String[0]), 0);
            }
        }
    }

    public static List<String> getAllPermission() {
        return Collections.unmodifiableList(Arrays.asList(ALL_PERMISSION));
    }

    private void showWaringDialog() {
        AlertDialog.Builder dialog = new AlertDialog.Builder(this);
        dialog.setMessage(R.string.app_need_permission)
                .setPositiveButton(R.string.app_permission_by_hand, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        //Guide the user to the setting page for manual authorization.
                        Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                        Uri uri = Uri.fromParts("package", getApplicationContext().getPackageName(), null);
                        intent.setData(uri);
                        startActivity(intent);
                    }
                })
                .setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        //Instruct the user to perform manual authorization. The permission request fails.
                        finish();
                    }
                }).setOnCancelListener(dialogInterface);
        dialog.setCancelable(false);
        dialog.show();
    }

    static DialogInterface.OnCancelListener dialogInterface = new DialogInterface.OnCancelListener() {
        @Override
        public void onCancel(DialogInterface dialog) {
            //Instruct the user to perform manual authorization. The permission request fails.
        }
    };
}