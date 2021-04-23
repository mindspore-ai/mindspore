package com.mindspore.hms.bonedetection;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.alibaba.android.arouter.launcher.ARouter;
import com.bumptech.glide.Glide;
import com.huawei.hmf.tasks.OnFailureListener;
import com.huawei.hmf.tasks.OnSuccessListener;
import com.huawei.hmf.tasks.Task;
import com.huawei.hms.mlsdk.common.MLFrame;
import com.huawei.hms.mlsdk.skeleton.MLSkeleton;
import com.huawei.hms.mlsdk.skeleton.MLSkeletonAnalyzer;
import com.huawei.hms.mlsdk.skeleton.MLSkeletonAnalyzerFactory;
import com.huawei.hms.mlsdk.skeleton.MLSkeletonAnalyzerSetting;
import com.mindspore.hms.BitmapUtils;
import com.mindspore.hms.R;
import com.mindspore.hms.camera.GraphicOverlay;
import com.mindspore.hms.camera.SkeletonGraphic;
import com.mindspore.hms.camera.SkeletonUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

@Route(path = "/hms/PosenetMainActivitys")
public class PosenetMainActivitys extends AppCompatActivity {
    private static final String TAG = "PosenetMainActivitys";
    private GraphicOverlay graphicOverlay;
    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int RC_CHOOSE_CAMERA = 2;

    private boolean isPreViewShow = false;

    private ImageView imgPreview;
    private Uri imageUri;

    private Bitmap originBitmap;
    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    private boolean isLandScape;
    private MLSkeletonAnalyzer analyzer;
    private MLFrame frame;
    private int analyzerType = -1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_posenet_main_activitys);
        init();
    }

    private void init() {
        graphicOverlay = findViewById(R.id.skeleton_previewOverlay);
        imgPreview = findViewById(R.id.img_origin);

        findViewById(R.id.jiance).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                originBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.skeleton_image);
                if (analyzerType != MLSkeletonAnalyzerSetting.TYPE_NORMAL || analyzer == null) {
                    stopAnalyzer();
                    createAnalyzer(MLSkeletonAnalyzerSetting.TYPE_NORMAL);
                    analyzerType = MLSkeletonAnalyzerSetting.TYPE_NORMAL;
                }
                showBonedetection();
            }
        });

        Toolbar mToolbar = findViewById(R.id.posenet_activity_toolbar);
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
    }

    /**
     * Create a skeleton analyzer
     *
     * @param analyzerType Normal or Yoga
     */
    private void createAnalyzer(int analyzerType) {
        MLSkeletonAnalyzerSetting setting = new MLSkeletonAnalyzerSetting.Factory()
                .setAnalyzerType(analyzerType)
                .create();
        analyzer = MLSkeletonAnalyzerFactory.getInstance().getSkeletonAnalyzer(setting);
    }

    public void onClickPhoto(View view) {
        openGallay(RC_CHOOSE_PHOTO);
    }

    public void onClickCamera(View view) {
        openCamera();
    }


    public void onClickRealTime(View view) {
        ARouter.getInstance().build("/posenet/LiveSkeletonAnalyseActivity").navigation();
    }

    private void openGallay(int request) {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, request);
    }

    private void openCamera() {
        Intent intentToTakePhoto = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        String mTempPhotoPath = Environment.getExternalStorageDirectory() + File.separator + "photo.jpeg";
        imageUri = FileProvider.getUriForFile(this, getApplicationContext().getPackageName() + ".fileprovider", new File(mTempPhotoPath));
        intentToTakePhoto.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        startActivityForResult(intentToTakePhoto, RC_CHOOSE_CAMERA);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (RC_CHOOSE_PHOTO == requestCode) {
                if (null != data && null != data.getData()) {
                    this.imageUri = data.getData();
                    showOriginImage();
                } else {
                    finish();
                }
            } else if (RC_CHOOSE_CAMERA == requestCode) {
                showOriginCamera();
            }
        }
    }

    private void showOriginImage() {
        File file = BitmapUtils.getFileFromMediaUri(this, imageUri);
        Bitmap photoBmp = BitmapUtils.getBitmapFormUri(this, Uri.fromFile(file));
        int degree = BitmapUtils.getBitmapDegree(file.getAbsolutePath());
        originBitmap = BitmapUtils.rotateBitmapByDegree(photoBmp, degree).copy(Bitmap.Config.ARGB_8888, true);
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
            isPreViewShow = true;
            if (analyzerType != MLSkeletonAnalyzerSetting.TYPE_NORMAL || analyzer == null) {
                stopAnalyzer();
                createAnalyzer(MLSkeletonAnalyzerSetting.TYPE_NORMAL);
                analyzerType = MLSkeletonAnalyzerSetting.TYPE_NORMAL;
            }
            showBonedetection();
        } else {
            isPreViewShow = false;
        }
    }

    private void createFrame() {

        // Gets the targeted width / height, only portrait.
        int maxHeight = ((View) imgPreview.getParent()).getHeight();
        int targetWidth = ((View) imgPreview.getParent()).getWidth();
        // Determine how much to scale down the image.
        float scaleFactor =
                Math.max(
                        (float) originBitmap.getWidth() / (float) targetWidth,
                        (float) originBitmap.getHeight() / (float) maxHeight);

        Bitmap resizedBitmap =
                Bitmap.createScaledBitmap(
                        originBitmap,
                        (int) (originBitmap.getWidth() / scaleFactor),
                        (int) (originBitmap.getHeight() / scaleFactor),
                        true);

        frame = new MLFrame.Creator().setBitmap(resizedBitmap).create();
    }

    private void showBonedetection() {
        createFrame();
        Task<List<MLSkeleton>> task = analyzer.asyncAnalyseFrame(frame);
        task.addOnSuccessListener(new OnSuccessListener<List<MLSkeleton>>() {
            @Override
            public void onSuccess(List<MLSkeleton> results) {

                // Detection success.
                List<MLSkeleton> skeletons = SkeletonUtils.getValidSkeletons(results);
                if (skeletons != null && !skeletons.isEmpty()) {
                    processSuccess(skeletons);
                } else {
                    processFailure("async analyzer result is null.");
                }

                // 检测成功。
                Toast.makeText(PosenetMainActivitys.this, "检测成功", Toast.LENGTH_SHORT).show();
            }
        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(Exception e) {
                // 检测失败。
                Toast.makeText(PosenetMainActivitys.this, "检测失败", Toast.LENGTH_SHORT).show();

                processFailure(e.getMessage());

            }
        });

    }
    private void processSuccess(List<MLSkeleton> results) {
        graphicOverlay.clear();
        SkeletonGraphic skeletonGraphic = new SkeletonGraphic(graphicOverlay, results);
        graphicOverlay.add(skeletonGraphic);
    }
    private void processFailure(String str) {
        Log.e(TAG, str);
    }

    private void showOriginCamera() {
        try {
            Pair<Integer, Integer> targetedSize = this.getTargetSize();
            int targetWidth = targetedSize.first;
            int maxHeight = targetedSize.second;
            Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
            originBitmap = BitmapUtils.zoomImage(bitmap, targetWidth, maxHeight);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        // Determine how much to scale down the image.
        Log.e(TAG, "resized image size width:" + originBitmap.getWidth() + ",height: " + originBitmap.getHeight());
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
            isPreViewShow = true;
            if (analyzerType != MLSkeletonAnalyzerSetting.TYPE_NORMAL || analyzer == null) {
                stopAnalyzer();
                createAnalyzer(MLSkeletonAnalyzerSetting.TYPE_NORMAL);
                analyzerType = MLSkeletonAnalyzerSetting.TYPE_NORMAL;
            }
            showBonedetection();
        } else {
            isPreViewShow = false;
        }
    }

    private Pair<Integer, Integer> getTargetSize() {
        Integer targetWidth;
        Integer targetHeight;
        Integer maxWidth = this.getMaxWidthOfImage();
        Integer maxHeight = this.getMaxHeightOfImage();
        targetWidth = this.isLandScape ? maxHeight : maxWidth;
        targetHeight = this.isLandScape ? maxWidth : maxHeight;
        Log.i(TAG, "height:" + targetHeight + ",width:" + targetWidth);
        return new Pair<>(targetWidth, targetHeight);
    }

    private Integer getMaxWidthOfImage() {
        if (this.maxWidthOfImage == null) {
            if (this.isLandScape) {
                this.maxWidthOfImage = ((View) this.imgPreview.getParent()).getHeight();
            } else {
                this.maxWidthOfImage = ((View) this.imgPreview.getParent()).getWidth();
            }
        }
        return this.maxWidthOfImage;
    }

    private Integer getMaxHeightOfImage() {
        if (this.maxHeightOfImage == null) {
            if (this.isLandScape) {
                this.maxHeightOfImage = ((View) this.imgPreview.getParent()).getWidth();
            } else {
                this.maxHeightOfImage = ((View) this.imgPreview.getParent()).getHeight();
            }
        }
        return this.maxHeightOfImage;
    }
    /**
     * Synchronous analyse.同步
     */
    /*private void analyzerSync() {
        List<MLSkeleton> list = new ArrayList<>();
        MLFrame frame = MLFrame.fromBitmap(originBitmap);
        SparseArray<MLSkeleton> sparseArray = analyzer.analyseFrame(frame);
        for (int i = 0; i < sparseArray.size(); i++) {
            list.add(sparseArray.get(i));
        }
        // Remove invalid point.
        List<MLSkeleton> skeletons = SkeletonUtils.getValidSkeletons(list);
        if (skeletons != null && !skeletons.isEmpty()) {
            processSuccess(skeletons);
        } else {
            processFailure("sync analyzer result is null.");
        }
    }*/

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopAnalyzer();
    }

    private void stopAnalyzer() {
        if (analyzer != null) {
            try {
                analyzer.stop();
            } catch (IOException e) {
                Log.e(TAG, "Failed to stop the analyzer: " + e.getMessage());
            }
        }
    }

}