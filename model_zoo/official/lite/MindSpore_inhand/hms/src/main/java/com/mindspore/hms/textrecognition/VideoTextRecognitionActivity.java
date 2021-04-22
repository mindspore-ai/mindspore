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
package com.mindspore.hms.textrecognition;

import android.content.res.Configuration;
import android.os.Bundle;
import android.view.SurfaceHolder;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.huawei.hms.mlsdk.common.LensEngine;
import com.huawei.hms.mlsdk.common.MLAnalyzer;
import com.huawei.hms.mlsdk.common.MLApplication;
import com.huawei.hms.mlsdk.text.MLText;
import com.huawei.hms.mlsdk.text.MLTextAnalyzer;
import com.mindspore.common.sp.Preferences;
import com.mindspore.hms.R;
import com.mindspore.hms.camera.GraphicOverlay;
import com.mindspore.hms.camera.LensEnginePreviewVideoText;
import com.mindspore.hms.camera.MLSenceTextDetectionGraphic;

import java.io.IOException;

@Route(path = "/hms/VideoTextRecognitionActivity")
public class VideoTextRecognitionActivity extends AppCompatActivity {
    private MLTextAnalyzer analyzer;
    private LensEngine mLensEngine;
    private final int lensType = LensEngine.FRONT_LENS;
    private LensEnginePreviewVideoText mPreview;
    private GraphicOverlay mGraphicOverlay;
    private MLSenceTextDetectionGraphic graphic;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_video_textrecognition);
        MLApplication.getInstance().setApiKey(Preferences.API_KEY);
        init();
    }

    private void init() {
        Toolbar mToolbar = findViewById(R.id.segmentation_toolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
        this.mPreview = this.findViewById(R.id.preview_video);
        this.mGraphicOverlay = this.findViewById(R.id.overlay);
        mPreview.getmSurfaceView().getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(@NonNull SurfaceHolder holder) {
                analyzer = new MLTextAnalyzer.Factory(VideoTextRecognitionActivity.this).setLanguage("zh").create();
                analyzer.setTransactor(new OcrDetectorProcessor(mGraphicOverlay));
                mLensEngine = new LensEngine.Creator(getApplicationContext(), analyzer)
                        .setLensType(LensEngine.BACK_LENS)
                        .applyDisplayDimension(640, 480)
                        .applyFps(25.0f)
                        .enableAutomaticFocus(true)
                        .create();
            }

            @Override
            public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {

                try {
                    mLensEngine.run(holder);

                } catch (IOException e) {
                    e.printStackTrace();
                }

            }

            @Override
            public void surfaceDestroyed(@NonNull SurfaceHolder holder) {

            }
        });
    }

    public class OcrDetectorProcessor implements MLAnalyzer.MLTransactor<MLText.Block> {
        private final GraphicOverlay mGraphicOverlay;

        OcrDetectorProcessor(GraphicOverlay ocrGraphicOverlay) {
            this.mGraphicOverlay = ocrGraphicOverlay;
        }

        @Override
        public void transactResult(MLAnalyzer.Result<MLText.Block> results) {

            if (isScreenChange()) {

            } else {
                mGraphicOverlay.clear();
                graphic = new MLSenceTextDetectionGraphic(mGraphicOverlay, results, VideoTextRecognitionActivity.this);
                mGraphicOverlay.add(graphic);
                mGraphicOverlay.postInvalidate();
            }


        }

        @Override
        public void destroy() {
            mGraphicOverlay.clear();
        }
    }

    public boolean isScreenChange() {
        Configuration mConfiguration = this.getResources().getConfiguration();
        int ori = mConfiguration.orientation;
        if (ori == Configuration.ORIENTATION_LANDSCAPE) {
            return true;
        } else if (ori == Configuration.ORIENTATION_PORTRAIT) {
            return false;
        }
        return false;
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

            }
        }
        if (mPreview != null) {
            mPreview.stop();
        }
    }
}