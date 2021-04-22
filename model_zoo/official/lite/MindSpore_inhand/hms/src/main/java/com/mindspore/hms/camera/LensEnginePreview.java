/*
 * Copyright (C) The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *   2019.12.15-Changed modify to use LensEngine
 *                  Huawei Technologies Co., Ltd.
 */

package com.mindspore.hms.camera;

import android.content.Context;
import android.content.res.Configuration;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.ViewGroup;

import com.huawei.hms.common.size.Size;
import com.huawei.hms.mlsdk.common.LensEngine;

import java.io.IOException;

public class LensEnginePreview extends ViewGroup {
    private static final String TAG = LensEnginePreview.class.getSimpleName();

    private final Context mContext;

    public SurfaceView getmSurfaceView() {
        return mSurfaceView;
    }

    private final SurfaceView mSurfaceView;

    private boolean mStartRequested;

    private boolean mSurfaceAvailable;

    private LensEngine mLensEngine;

    private GraphicOverlay mOverlay;

    public LensEnginePreview(Context context, AttributeSet attrs) {
        super(context, attrs);
        this.mContext = context;
        this.mStartRequested = false;
        this.mSurfaceAvailable = false;

        this.mSurfaceView = new SurfaceView(context);
        this.mSurfaceView.getHolder().addCallback(new SurfaceCallback());
        this.addView(this.mSurfaceView);
    }

    public void start(LensEngine lensEngine) throws IOException {
        if (lensEngine == null) {
            this.stop();
        }

        this.mLensEngine = lensEngine;

        if (this.mLensEngine != null) {
            this.mStartRequested = true;
            this.startIfReady();
        }
    }

    public void start(LensEngine lensEngine, GraphicOverlay overlay) throws IOException {
        this.mOverlay = overlay;
        this.start(lensEngine);
    }

    public void stop() {
        if (this.mLensEngine != null) {
            this.mLensEngine.close();
        }
    }

    public void release() {
        if (this.mLensEngine != null) {
            this.mLensEngine.release();
            this.mLensEngine = null;
        }
    }

    private void startIfReady() throws IOException {
        if (this.mStartRequested && this.mSurfaceAvailable) {
            this.mLensEngine.run(this.mSurfaceView.getHolder());
            if (this.mOverlay != null) {
                Size size = this.mLensEngine.getDisplayDimension();
                int min = Math.min(size.getWidth(), size.getHeight());
                int max = Math.max(size.getWidth(), size.getHeight());
                if (this.isPortraitMode()) {
                    // Swap width and height sizes when in portrait, since it will be rotated by 90 degrees.
                    this.mOverlay.setCameraInfo(min, max, this.mLensEngine.getLensType());
                } else {
                    this.mOverlay.setCameraInfo(max, min, this.mLensEngine.getLensType());
                }
                this.mOverlay.clear();
            }
            this.mStartRequested = false;
        }
    }

    private class SurfaceCallback implements SurfaceHolder.Callback {
        @Override
        public void surfaceCreated(SurfaceHolder surface) {
            LensEnginePreview.this.mSurfaceAvailable = true;
            try {
                LensEnginePreview.this.startIfReady();
            } catch (IOException e) {
                Log.e(LensEnginePreview.TAG, "Could not start camera source.", e);
            }
        }

        @Override
        public void surfaceDestroyed(SurfaceHolder surface) {
            LensEnginePreview.this.mSurfaceAvailable = false;
        }

        @Override
        public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        }
    }

    @Override
    protected void onLayout(boolean changed, int left, int top, int right, int bottom) {
        int previewWidth = 480;
        int previewHeight = 360;
        if (this.mLensEngine != null) {
            Size size = this.mLensEngine.getDisplayDimension();
            if (size != null) {
                previewWidth = size.getWidth();
                previewHeight = size.getHeight();
            }
        }

        // Swap width and height sizes when in portrait, since it will be rotated 90 degrees
        if (this.isPortraitMode()) {
            int tmp = previewWidth;
            previewWidth = previewHeight;
            previewHeight = tmp;
        }

        final int viewWidth = right - left;
        final int viewHeight = bottom - top;

        int childWidth;
        int childHeight;
        int childXOffset = 0;
        int childYOffset = 0;
        float widthRatio = (float) viewWidth / (float) previewWidth;
        float heightRatio = (float) viewHeight / (float) previewHeight;

        // To fill the view with the camera preview, while also preserving the correct aspect ratio,
        // it is usually necessary to slightly oversize the child and to crop off portions along one
        // of the dimensions. We scale up based on the dimension requiring the most correction, and
        // compute a crop offset for the other dimension.
        if (widthRatio > heightRatio) {
            childWidth = viewWidth;
            childHeight = (int) ((float) previewHeight * widthRatio);
            childYOffset = (childHeight - viewHeight) / 2;
        } else {
            childWidth = (int) ((float) previewWidth * heightRatio);
            childHeight = viewHeight;
            childXOffset = (childWidth - viewWidth) / 2;
        }

        for (int i = 0; i < this.getChildCount(); ++i) {
            // One dimension will be cropped. We shift child over or up by this offset and adjust
            // the size to maintain the proper aspect ratio.
            this.getChildAt(i).layout(-1 * childXOffset, -1 * childYOffset, childWidth - childXOffset,
                    childHeight - childYOffset);
        }

        try {
            this.startIfReady();
        } catch (IOException e) {
            Log.e(LensEnginePreview.TAG, "Could not start camera source.", e);
        }
    }

    private boolean isPortraitMode() {
        int orientation = this.mContext.getResources().getConfiguration().orientation;
        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
            return false;
        }
        if (orientation == Configuration.ORIENTATION_PORTRAIT) {
            return true;
        }

        Log.d(LensEnginePreview.TAG, "isPortraitMode returning false by default");
        return false;
    }
}
