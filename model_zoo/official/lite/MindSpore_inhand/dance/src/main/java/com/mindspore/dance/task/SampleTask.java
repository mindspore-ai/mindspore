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

package com.mindspore.dance.task;

import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.camera.view.PreviewView;

import com.huawei.hmf.tasks.OnFailureListener;
import com.huawei.hmf.tasks.Task;
import com.huawei.hms.mlsdk.common.MLFrame;
import com.huawei.hms.mlsdk.skeleton.MLSkeleton;
import com.huawei.hms.mlsdk.skeleton.MLSkeletonAnalyzer;
import com.huawei.hms.mlsdk.skeleton.MLSkeletonAnalyzerFactory;
import com.mindspore.dance.algorithm.ModelDataBean;
import com.mindspore.dance.algorithm.ModelDataUtils;
import com.mindspore.dance.global.Variables;

import java.util.ArrayList;
import java.util.List;

public class SampleTask implements Runnable {
    private static final String TAG = SampleTask.class.getSimpleName();
    private PreviewView previewView;
    private MLSkeletonAnalyzer analyzer;
    private List<List<MLSkeleton>> bitmapsSkeleton;
    private List<ModelDataBean> modelDataBeanList;

    public SampleTask(PreviewView previewView) {
        this.previewView = previewView;
        init();
    }

    private void init() {
        analyzer = MLSkeletonAnalyzerFactory.getInstance().getSkeletonAnalyzer();
        bitmapsSkeleton = new ArrayList<List<MLSkeleton>>();
        modelDataBeanList = new ArrayList<ModelDataBean>();
    }

    public void setNeedStop(boolean needStop) {
        this.needStop = needStop;

    }

    public void clear() {
        bitmapsSkeleton.clear();
        analyzer = null;
        previewView = null;
    }

    private int num = 0;
    private boolean needStop = false;

    @Override
    public void run() {
        while ((!needStop) && num < 70) {
            sample();
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        double score = ModelDataUtils.getScore(modelDataBeanList);
        Log.d(TAG, "getScore:" + score);
        Variables.score = (int) score;
    }

    private void getBitmap() {
        Handler mainHandler = new Handler(Looper.getMainLooper());

        mainHandler.post(() -> {
            if (previewView == null) {
                return;
            }
            Bitmap tempBitmap = previewView.getBitmap();
            if (tempBitmap == null) {
                modelDataBeanList.add(null);
                Log.w(TAG, "bitmap is null.");
                return;
            }

            MLFrame frame = MLFrame.fromBitmap(tempBitmap);
            try {
                Task<List<MLSkeleton>> task = analyzer.asyncAnalyseFrame(frame);
                task.addOnSuccessListener(results -> {
                    // check successful
                    if (results.size() > 0) {
                        Log.d(TAG, "onSuccess " + "skeleton:" + results.get(0));
                        modelDataBeanList.add(ModelDataUtils.hmsData2ModelData(results.get(0)));
                    } else {
                        Log.w(TAG, "onSuccess, but no have skeleton in bitmap.");
                        modelDataBeanList.add(null);
                    }
                }).addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(Exception e) {
                        // check failed.
                        Log.d(TAG, "onFailure sample:" + num);
                    }
                });
            } catch (IllegalArgumentException e) {
                Log.d(TAG, "getBitmap," + e.getMessage());
            } catch (IllegalStateException e) {
                Log.d(TAG, "getBitmap," + e.getMessage());
            }
        });
    }

    private void sample() {
        Log.d(TAG, "sample:" + num);
        if (previewView != null) {
            getBitmap();
        }
        num++;
    }
}
