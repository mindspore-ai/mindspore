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

package com.mindspore.dance.present.video;

import android.content.Context;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.widget.MediaController;
import android.widget.TextView;
import android.widget.VideoView;

import com.mindspore.dance.R;
import com.mindspore.dance.task.GoneViewTask;

import java.util.Formatter;
import java.util.Locale;

public class MyVideoView extends VideoView {
    private static final String TAG = MyVideoView.class.getSimpleName();
    private MediaController mediaController;
    private Context mContext;
    private TextView mCountdownView;

    public MyVideoView(Context context) {
        super(context);
        mContext = context;
        mediaController = new MediaController(context);
        this.setMediaController(mediaController);
        initMediaController();
    }

    public void setCountdownView(TextView countdownView) {
        mCountdownView = countdownView;

    }

    private void initMediaController() {
        mediaController.setEnabled(false);
    }

    @Override
    public void pause() {
        super.pause();
        Log.d(TAG, "pause");
    }

    @Override
    public void resume() {
        super.resume();
        Log.d(TAG, "resume");
    }

    @Override
    public void start() {
        super.start();
        Log.d(TAG, "start");
    }

    @Override
    public boolean onTouchEvent(MotionEvent ev) {
        Log.d(TAG, "onTouchEvent");
        int remainingTime = this.getDuration() - this.getCurrentPosition();
        String timeStr = stringForTime(remainingTime);
        if (mCountdownView != null) {
            if (mCountdownView.getVisibility() == GONE) {
                mCountdownView.setText(getContext().getString(R.string.countdown) + timeStr);
                mCountdownView.setVisibility(VISIBLE);
                new Thread(new GoneViewTask(mCountdownView)).start();
            } else {
                mCountdownView.setVisibility(GONE);
            }
        }
        return false;
    }

    private String stringForTime(int timeMs) {
        if (timeMs < 0) {
            return "00:00";
        }
        StringBuilder mFormatBuilder = new StringBuilder();
        Formatter mFormatter = new Formatter(mFormatBuilder, Locale.getDefault());
        int totalSeconds = timeMs / 1000;

        int seconds = totalSeconds % 60;
        int minutes = (totalSeconds / 60) % 60;
        int hours = totalSeconds / 3600;

        mFormatBuilder.setLength(0);
        if (hours > 0) {
            return mFormatter.format("%d:%02d:%02d", hours, minutes, seconds).toString();
        } else {
            return mFormatter.format("%02d:%02d", minutes, seconds).toString();
        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        return false;
    }

}
