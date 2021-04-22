package com.mindspore.hms.ImageSegmentation;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.SparseArray;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.bumptech.glide.Glide;
import com.huawei.hms.mlsdk.MLAnalyzerFactory;
import com.huawei.hms.mlsdk.common.LensEngine;
import com.huawei.hms.mlsdk.common.MLAnalyzer;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentation;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentationAnalyzer;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentationScene;
import com.huawei.hms.mlsdk.imgseg.MLImageSegmentationSetting;
import com.mindspore.common.utils.ImageUtils;
import com.mindspore.hms.ImageSegmentation.overlay.CameraImageGraphic;
import com.mindspore.hms.ImageSegmentation.overlay.GraphicOverlay;
import com.mindspore.hms.R;
import com.mindspore.hms.camera.LensEnginePreview;

import java.io.IOException;

@Route(path = "/hms/PortraitSegmentationActivity")
public class PortraitSegmentationActivity extends AppCompatActivity implements OnBackgroundImageListener {

    private static final int[] IMAGES = {R.drawable.portrait1, R.drawable.portrait2, R.drawable.portrait3, R.drawable.portrait4, R.drawable.portrait5, R.drawable.icon_default};

    private RecyclerView recyclerView;

    private MLImageSegmentationAnalyzer analyzer;
    private LensEnginePreview mPreview;
    private GraphicOverlay mOverlay;
    private LensEngine mLensEngine;

    private int lensType = LensEngine.FRONT_LENS;
    private boolean isFront = true;

    private ImageView background;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_portrait);

        init();
        createSegmentAnalyzer();
        if (savedInstanceState != null) {
            this.lensType = savedInstanceState.getInt("lensType");
        }
    }

    private void init() {
        Toolbar mToolbar = findViewById(R.id.activity_toolbar);
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());

        recyclerView = findViewById(R.id.recycleView);
        LinearLayoutManager mHorizontalLinearLayout = new LinearLayoutManager(this, RecyclerView.HORIZONTAL, false);
        recyclerView.setLayoutManager(mHorizontalLinearLayout);
        recyclerView.setAdapter(new PortraitSegmentationAdapter(this, IMAGES, this));

        mPreview = findViewById(R.id.preview);
        mOverlay = findViewById(R.id.graphic);
        background = findViewById(R.id.background);
    }

    @Override
    public void onSaveInstanceState(Bundle savedInstanceState) {
        savedInstanceState.putInt("lensType", this.lensType);
        super.onSaveInstanceState(savedInstanceState);
    }

    @Override
    protected void onResume() {
        super.onResume();
        this.createLensEngine();
        this.startLensEngine();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_setting, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int itemId = item.getItemId();
        if (itemId == R.id.item_camera) {
            isFront = !isFront;
            if (isFront) {
                lensType = LensEngine.FRONT_LENS;
            } else {
                lensType = LensEngine.BACK_LENS;
            }
            if (mLensEngine != null) {
                mLensEngine.close();
            }
            createLensEngine();
            startLensEngine();
        } else if (itemId == R.id.item_save) {

            mOverlay.setTag("mOverlay");
            mPreview.setTag("mPreview");
            Uri imgPath = ImageUtils.saveToAlbum(PortraitSegmentationActivity.this, mPreview,null,true);
            if (imgPath != null) {
                Toast.makeText(PortraitSegmentationActivity.this,"保存成功", Toast.LENGTH_SHORT).show();
            }
        }
        return super.onOptionsItemSelected(item);
    }

    private void createSegmentAnalyzer() {
        MLImageSegmentationSetting setting = new MLImageSegmentationSetting.Factory()
                .setExact(false)
                .setScene(MLImageSegmentationScene.FOREGROUND_ONLY)
                .setAnalyzerType(MLImageSegmentationSetting.BODY_SEG)
                .create();
        this.analyzer = MLAnalyzerFactory.getInstance().getImageSegmentationAnalyzer(setting);
        this.analyzer.setTransactor(new MLAnalyzer.MLTransactor<MLImageSegmentation>() {
            @Override
            public void destroy() {

            }

            @Override
            public void transactResult(MLAnalyzer.Result<MLImageSegmentation> result) {
                SparseArray<MLImageSegmentation> imageSegmentationResult = result.getAnalyseList();
                Bitmap bitmap = imageSegmentationResult.valueAt(0).getForeground();
                if (isFront) {
                    bitmap = convert(bitmap);
                }
                mOverlay.clear();
                CameraImageGraphic cameraImageGraphic = new CameraImageGraphic(mOverlay, bitmap);
                mOverlay.addGraphic(cameraImageGraphic);
                mOverlay.postInvalidate();
            }
        });
    }

    private void createLensEngine() {
        Context context = this.getApplicationContext();
        // Create LensEngine.
        this.mLensEngine = new LensEngine.Creator(context, this.analyzer)
                .setLensType(lensType)
                .applyDisplayDimension(1280, 720)
                .applyFps(25.0f)
                .enableAutomaticFocus(true)
                .create();
    }


    private void startLensEngine() {
        if (mLensEngine != null) {
            try {
                mPreview.start(mLensEngine);
            } catch (IOException e) {
                mLensEngine.release();
                mLensEngine = null;
            }
        }
    }

    private Bitmap convert(Bitmap bitmap) {
        Matrix m = new Matrix();
        m.setScale(-1, 1);
        Bitmap reverseBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), m, true);
        return reverseBitmap;
    }

    public void stopPreview() {
        mPreview.stop();
    }


    @Override
    public void onBackImageSelected(int position) {
        Glide.with(this).load(IMAGES[position]).into(background);
    }

    @Override
    public void onImageAdd(View view) {
        openGallay(1000);
    }

    private void openGallay(int request) {
        Intent intentToPickPic = new Intent(Intent.ACTION_PICK, null);
        intentToPickPic.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intentToPickPic, 1000);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (1000 == requestCode) {
                if (null != data && null != data.getData()) {
                    Glide.with(this).load(data.getData()).into(background);
                }
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (analyzer != null) {
            try {
                analyzer.stop();
            } catch (IOException e) {
            }
        }
        if (mLensEngine != null) {
            mLensEngine.release();
        }
    }
}