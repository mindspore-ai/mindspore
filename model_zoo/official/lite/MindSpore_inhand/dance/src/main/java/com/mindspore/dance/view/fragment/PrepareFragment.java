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

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.MediaController;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.navigation.fragment.NavHostFragment;

import com.mindspore.common.base.mvp.BaseFragment;
import com.mindspore.common.net.FileDownLoadObserver;
import com.mindspore.dance.R;
import com.mindspore.dance.global.Constants;
import com.mindspore.dance.global.Variables;
import com.mindspore.dance.present.video.MyVideoView;
import com.mindspore.dance.util.Tools;
import com.mindspore.dance.view.mvp.PrepareContract;
import com.mindspore.dance.view.mvp.PreparePresenter;

import java.io.File;

import static com.mindspore.common.utils.Utils.getApp;

public class PrepareFragment extends BaseFragment<PreparePresenter> implements PrepareContract.View {
    private final String TAG = PrepareFragment.class.getSimpleName();

    private FrameLayout root;
    private TextView downloadText;
    private MyVideoView videoView;
    private MediaController mediaController;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        presenter = new PreparePresenter(this);

        Tools.verifyStoragePermissions(getActivity());
        Tools.verifyCameraPermissions(getActivity());
        Tools.checkDiskHasVideo();
        if (!Variables.hasVideo) {
            downFile();
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_prepare, container, false);
    }

    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        root = view.findViewById(R.id.root);
        downloadText = view.findViewById(R.id.download_text);
        initView();
    }

    public void downFile() {
        presenter.downloadDanceVideo(new FileDownLoadObserver<File>() {
            @Override
            public void onDownLoadSuccess(File file) {
                Tools.checkDiskHasVideo();
                downloadText.setVisibility(View.GONE);
                if (videoView != null && !videoView.isPlaying()) {
                    start();
                }
            }

            @Override
            public void onDownLoadFail(Throwable throwable) {
                Toast.makeText(getApp(), R.string.download_faild, Toast.LENGTH_SHORT).show();

            }

            @Override
            public void onProgress(final int progress, long total) {
                getActivity().runOnUiThread(() -> {
                    downloadText.setText(String.format(getString(R.string.downloading), progress, "%"));
                });
            }
        });
    }


    @Override
    public void onResume() {
        super.onResume();
        if (!Variables.hasVideo) {
            downFile();
        } else {
            if (videoView != null && !videoView.isPlaying()) {
                videoView.start();
            }
        }
    }

    @Override
    public void onStop() {
        super.onStop();
        if (videoView != null && videoView.isPlaying()) {
            videoView.pause();
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        onStop();
        videoView = null;
    }

    private void initView() {
        addVideoView();
        addBackground();
        addButton();
        if (!Variables.hasVideo) {
            downloadText.setVisibility(View.VISIBLE);
        }
    }

    private void addVideoView() {
        videoView = new MyVideoView(getActivity());
        Tools.addViewFixedScale(root, videoView, 0.625f, 0.596f, 16f / 9f,
                0.227f, 0.187f, Tools.TYPE_VIDEO_VIEW);

        videoView.setOnCompletionListener(mp -> videoView.start());
        if (Variables.hasVideo) {
            start();
        }
    }

    private void addBackground() {
        ImageView backgroundView = new ImageView(getContext());
        backgroundView.setBackgroundResource(R.drawable.prepare_bk);
        Tools.addView(root, backgroundView, 1f, 1f, -1, -1);
    }

    private void addButton() {
        ImageButton buttonView = new ImageButton(getContext());
        buttonView.setBackgroundResource(R.drawable.begin_run_bt);
        buttonView.setOnClickListener(v -> {
            int permission = ActivityCompat.checkSelfPermission(getActivity(), Manifest.permission.CAMERA);
            if (permission != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(getActivity(), R.string.toast_no_camera_permission, Toast.LENGTH_SHORT).show();
                return;
            }
            permission = ActivityCompat.checkSelfPermission(getActivity(), Manifest.permission.WRITE_EXTERNAL_STORAGE);
            if (permission != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(getActivity(), R.string.toast_no_storage_permission,
                        Toast.LENGTH_SHORT).show();
                return;
            }
            if (!Variables.hasVideo) {
                return;
            }
            NavHostFragment.findNavController(PrepareFragment.this)
                    .navigate(R.id.action_PrepareFragment_to_RunFragment);
        });
        Tools.addView(root, buttonView, 0.256f, 0.132f, 0.843f, 0.372f);

        ImageButton backButton = new ImageButton(getContext());
        backButton.setBackgroundResource(R.drawable.dance_back);
        backButton.setOnClickListener(view -> {
            getActivity().finish();
        });
        Tools.addView(root, backButton, 0.1f, 0.2f, 0.2f, 0.06f);
    }

    private void start() {
        mediaController = new MediaController(getContext());
        String uri = Constants.VIDEO_PATH + Constants.VIDEO_NAME;
        Log.d(TAG, "start uri:" + uri);
        videoView.setVideoPath(uri);
        videoView.setMediaController(mediaController);
        mediaController.setMediaPlayer(videoView);
        videoView.requestFocus();
        videoView.start();
    }

}