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

package com.mindspore.dance.util;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Rect;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;

import androidx.annotation.MainThread;
import androidx.core.app.ActivityCompat;

import com.mindspore.dance.global.Constants;
import com.mindspore.dance.global.Variables;

import java.io.File;

public class Tools {
    private static String TAG = Tools.class.getSimpleName();
    public static int WIDTH_INDEX = 0;
    public static int HEIGHT_INDEX = 1;
    public static final int TYPE_VIDEO_VIEW = 0;
    public static final int TYPE_CAMERA_VIEW = 1;

    /**
     * Get the download status, get the width and height of a view, (There is a problem, you may get {0,0})
     *
     * @param view
     * @return int[2]  int[0] = view.width; int[1] = view.height;
     */
    public static int[] getWH(View view) {
        int w = View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED);
        int h = View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED);

        view.measure(w, h);
        int width = view.getMeasuredWidth();
        int height = view.getMeasuredHeight();
        Log.d(TAG, "rootLayout width:" + width + ", height:" + height);
        int[] a = {width, height};
        return a;
    }

    /**
     * Get the width and height of an activity (including the status bar)
     *
     * @return int[2] int[0] = width; int[1] = height;
     */
    public static int[] getActivityWH(Activity activity) {
        Display display = activity.getWindowManager().getDefaultDisplay();
        Log.d(TAG, "Activity width:" + display.getWidth() + ", height:" + display.getHeight());
        int[] a = {display.getWidth(), display.getHeight()};
        return a;
    }

    /**
     * Get the visual area of a view
     *
     * @param view
     * @return int[2]  int[0] = width; int[1] = height;
     */
    public static int[] getRect(View view) {
        Rect rectangle = new Rect();
        view.getWindowVisibleDisplayFrame(rectangle);
        Log.d(TAG, "getRect parentView width:" + rectangle.width() + ", height:" + rectangle.height());
        int[] a = {rectangle.width(), rectangle.height()};
        return a;
    }

    /**
     * Adding childView to parentView must be done in the main thread.
     *
     * @param parentView      Parent container
     * @param childView       Control to add
     * @param widthRatio      Control width / parent container width
     * @param highRatio       Control height / parent container height
     *                        if leftMarginRatio = -1; the horizontal center; topMarginRatio = -1 Indicates vertical center
     * @param topMarginRatio  Control top margin / parent container height
     * @param leftMarginRatio Control left margin / parent container width
     */
    @MainThread
    public static void addView(ViewGroup parentView, View childView, float widthRatio, float highRatio,
                               float topMarginRatio, float leftMarginRatio) {
        int[] parentMeasure = Tools.getRect(parentView);
        int width = (int) (parentMeasure[0] * widthRatio);
        int high = (int) (parentMeasure[1] * highRatio);
        int left = leftMarginRatio == -1 ? (parentMeasure[0] - width) / 2
                : (int) (parentMeasure[0] * leftMarginRatio);
        int top = topMarginRatio == -1 ? (parentMeasure[1] - high) / 2
                : (int) (parentMeasure[1] * topMarginRatio);
        Log.d(TAG, "addView width:" + width + ", height:" + high + ", left:" + left + ", top:" + top);
        FrameLayout.LayoutParams fLayoutParams = new FrameLayout.LayoutParams(width, high);
        fLayoutParams.setMargins(left, top, 0, 0);

        childView.setLayoutParams(fLayoutParams);
        parentView.addView(childView);
    }

    /**
     * Adding childView to parentView must be done in the main thread.
     *
     * @param parentView      Parent container
     * @param childView       Control to add
     * @param widthRatio      Control width / parent container width
     * @param highRatio       Control height / parent container height
     * @param widthHighRatio  Control width / control height
     *                        if leftMarginRatio = -1; the horizontal center; topMarginRatio = -1 Indicates vertical center
     * @param topMarginRatio  Control top margin / parent container height
     * @param leftMarginRatio Control left margin / parent container width
     * @param type            Control is video 0 , a camera 1
     */
    @MainThread
    public static void addViewFixedScale(ViewGroup parentView, View childView, float widthRatio,
                                         float highRatio, float widthHighRatio,
                                         float topMarginRatio, float leftMarginRatio, int type) {
        if (widthHighRatio == 0) {
            return;
        }
        int[] parentMeasure = Tools.getRect(parentView);
        int width = (int) (parentMeasure[0] * widthRatio);
        int highBorder = (int) (parentMeasure[1] * highRatio);
        int high = (int) ((float) width / widthHighRatio);
        int left = leftMarginRatio == -1 ? (parentMeasure[0] - width) / 2
                : (int) (parentMeasure[0] * leftMarginRatio);
        int top = topMarginRatio == -1 ? (parentMeasure[1] - high) / 2
                : (int) (parentMeasure[1] * topMarginRatio);
        Log.d(TAG, "addViewFixedScale width:" + width + ", height:" + high + ", highBorder:" + highBorder + ", left:" + left + ", top:" + top);
        FrameLayout.LayoutParams fLayoutParams = new FrameLayout.LayoutParams(width, high);
        int adjustment = 0;
        if (high != highBorder) {
            switch (type) {
                case Tools.TYPE_VIDEO_VIEW:
                    adjustment = (highBorder - high) / 2;
                    break;
                case Tools.TYPE_CAMERA_VIEW:
                    adjustment = (highBorder - high);
                    break;
                default:
                    Log.e(TAG, "type is invalid. don't adjust.");
            }
        }
        Log.e(TAG, "addViewFixedScale adjustment." + adjustment);
        fLayoutParams.setMargins(left, top + adjustment, 0, 0);

        childView.setLayoutParams(fLayoutParams);
        parentView.addView(childView);
    }

    public static boolean checkDiskHasVideo() {
        File f = new File(Constants.VIDEO_PATH + Constants.VIDEO_NAME);
        if (f.exists() && f.length() == Constants.VIDEO_LENGTH) {
            Variables.hasVideo = true;
        } else {
            Variables.hasVideo = false;
        }
        return Variables.hasVideo;
    }

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static final int REQUEST_CAMERA = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    private static String[] PERMISSIONS_CAMERA = {
            Manifest.permission.CAMERA
    };

    public static void verifyStoragePermissions(Activity activity) {
        // Check if we have write permission
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE, REQUEST_EXTERNAL_STORAGE);
        }
    }

    public static void verifyCameraPermissions(Activity activity) {
        // Check if we have write permission
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.CAMERA);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(activity, PERMISSIONS_CAMERA, REQUEST_CAMERA);
        }
    }

}
