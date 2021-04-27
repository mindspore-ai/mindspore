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

package com.mindspore.dance.view.fragment;

import android.annotation.SuppressLint;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.MediaController;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.camera.camera2.Camera2Config;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.CameraXConfig;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.LifecycleOwner;
import androidx.navigation.fragment.NavHostFragment;

import com.google.common.util.concurrent.ListenableFuture;
import com.mindspore.dance.R;
import com.mindspore.dance.global.Constants;
import com.mindspore.dance.present.video.MyVideoView;
import com.mindspore.dance.task.SampleTask;
import com.mindspore.dance.util.Tools;

import java.util.concurrent.ExecutionException;

public class RunFragment extends Fragment {
    private final String TAG = RunFragment.class.getSimpleName();
    private FrameLayout root;
    private MyVideoView videoView;
    private PreviewView cameraView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private SampleTask sampleTask;
    private TextView countdownView;
    private ImageView info;
    private Handler handler;
    private final int NEED_CANCEL = -1;
    private MediaController mediaController;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_run, container, false);
    }

    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        root = view.findViewById(R.id.root);
        handler = new Handler();
        mediaController = new MediaController(this.getContext());
        initView();

        cameraProviderFuture = ProcessCameraProvider.getInstance(getContext());
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                // No errors need to be handled for this Future.
                // This should never be reached.
            }
        }, ContextCompat.getMainExecutor(getContext()));
    }

    private void complete() {
        if (videoView != null && videoView.isPlaying()) {
            videoView.stopPlayback();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.P)
    @Override
    public void onResume() {
        info.setVisibility(View.VISIBLE);
        info.setBackgroundResource(R.drawable.countdown_three);
        super.onResume();
        handler.postDelayed(() -> {
            //execute the task
            info.setVisibility(View.VISIBLE);
            info.setBackgroundResource(R.drawable.countdown_two);
        }, NEED_CANCEL, 1000);
        handler.postDelayed(() -> {
            //execute the task
            info.setVisibility(View.VISIBLE);
            info.setBackgroundResource(R.drawable.countdown_one);
        }, NEED_CANCEL, 2000);
        handler.postDelayed(() -> {
            //execute the task
            beginPlay();
            info.setVisibility(View.GONE);
        }, NEED_CANCEL, 3080);
    }

    private void start() {
        String uri = Constants.VIDEO_PATH + Constants.VIDEO_NAME;
        Log.d(TAG, "start uri:" + uri);
        videoView.setVideoPath(uri);
        videoView.setMediaController(mediaController);
        mediaController.setMediaPlayer(videoView);
        videoView.requestFocus();
        videoView.start();
    }

    public void beginPlay() {
        addButtonView();
        start();
        sampleTask = new SampleTask(cameraView);
        new Thread(sampleTask).start();
    }

    @Override
    public void onPause() {
        super.onPause();
        handler.removeMessages(NEED_CANCEL);
        if (sampleTask != null) {
            sampleTask.setNeedStop(true);
            sampleTask.clear();
            sampleTask = null;
        }
        if (videoView.isPlaying()) {
            videoView.stopPlayback();
        }
    }

    @SuppressLint("RestrictedApi")
    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT).build();

        preview.setSurfaceProvider(cameraView.getSurfaceProvider());
        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview);
    }

    private void initView() {
        addVideoView();
        addTextInfoView();
        addCameraView();
        addCountdownView();
        addBackground();

    }

    private void addTextInfoView() {
        info = new ImageView(getActivity());
        info.setBackgroundResource(R.drawable.countdown_three);
        Tools.addView(root, info, 0.872f, 0.875f, 0.091f, 0.062f);
    }

    private void addCountdownView() {
        countdownView = new TextView(getContext());
        countdownView.setTextSize(20);
        countdownView.setTextColor(0xFFFFFFFF);
        countdownView.setBackgroundColor(0xff333333);
        countdownView.setGravity(Gravity.CENTER);
        Tools.addView(root, countdownView, 0.256f, 0.113f, 0.091f, 0.062f);
        if (videoView != null) {
            videoView.setCountdownView(countdownView);
        }
        countdownView.setVisibility(View.GONE);
    }

    private void addBackground() {
        ImageView backgroundView = new ImageView(getContext());
        backgroundView.setBackgroundResource(R.drawable.run_bk);
        Tools.addView(root, backgroundView, 1f, 1f, -1, -1);
    }

    private void addVideoView() {
        videoView = new MyVideoView(getActivity());
        Tools.addViewFixedScale(root, videoView, 0.872f, 0.875f, 16f / 9f,
                0.091f, 0.062f, Tools.TYPE_VIDEO_VIEW);
        videoView.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
            @Override
            public void onCompletion(MediaPlayer mp) {
                complete();
                NavHostFragment.findNavController(RunFragment.this)
                        .navigate(R.id.action_RunFragment_to_ResultFragment);
            }
        });
    }

    private void addCameraView() {
        cameraView = new PreviewView(getContext());
        cameraView.setScaleType(PreviewView.ScaleType.FIT_CENTER);
        Tools.addViewFixedScale(root, cameraView, 0.248f, 0.331f, 4f / 3f,
                0.635f, 0.686f, Tools.TYPE_CAMERA_VIEW);
    }

    private void addButtonView() {
        ImageButton backButton = new ImageButton(getContext());
        backButton.setBackgroundResource(R.drawable.dance_back);
        backButton.setOnClickListener(view -> {
            getActivity().finish();
        });
        Tools.addView(root, backButton, 0.1f, 0.2f, 0.12f, 0.1f);
    }
}
