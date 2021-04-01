/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
package com.mindspore.himindspore.ui.guide;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.net.Uri;
import android.preference.PreferenceManager;
import android.provider.Settings;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.TextPaint;
import android.text.method.LinkMovementMethod;
import android.text.style.ClickableSpan;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.WindowManager;
import android.widget.PopupWindow;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;

import com.mindspore.common.sp.Preferences;
import com.mindspore.common.utils.Utils;
import com.mindspore.customview.countdown.MSCountDownView;
import com.mindspore.himindspore.R;
import com.mindspore.himindspore.base.BaseActivity;
import com.mindspore.himindspore.ui.main.MainActivity;
import com.mindspore.himindspore.ui.main.PrivacyPolicyActivity;

import java.util.List;

import pub.devrel.easypermissions.AfterPermissionGranted;
import pub.devrel.easypermissions.AppSettingsDialog;
import pub.devrel.easypermissions.EasyPermissions;


public class SplashActivity extends BaseActivity implements EasyPermissions.PermissionCallbacks {

    private static final String TAG = "SplashActivity";

    private static final String[] PERMISSIONS = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_PHONE_STATE, Manifest.permission.CAMERA};
    private static final int REQUEST_PERMISSION = 1;

    private SharedPreferences prefs;

    private MSCountDownView cdvTime;

    private boolean isCheckPrivacy = false;
    private View mContentView;
    private TextView mTv_protocol;
    private PopupWindow mPopupW;

    @Override
    protected void init() {
        cdvTime = findViewById(R.id.cdv_time);
        prefs = PreferenceManager.getDefaultSharedPreferences(Utils.getApp());
        initCountDownView();
    }

    private void initCountDownView() {
        cdvTime.setTime(3);
        cdvTime.start();
        cdvTime.setOnLoadingFinishListener(() -> check());
        cdvTime.setOnClickListener(view -> {
            cdvTime.stop();
            check();
        });
    }

    @Override
    public int getLayout() {
        return R.layout.activity_splash;
    }


    @AfterPermissionGranted(REQUEST_PERMISSION)
    private void startPermissionsTask() {
        if (hasPermissions()) {
            setHandler();
        } else {
            EasyPermissions.requestPermissions(this,
                    this.getResources().getString(R.string.app_need_permission),
                    REQUEST_PERMISSION, PERMISSIONS);
        }
    }

    private boolean hasPermissions() {
        return EasyPermissions.hasPermissions(this, PERMISSIONS);
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        EasyPermissions.onRequestPermissionsResult(requestCode,
                permissions, grantResults, SplashActivity.this);
    }

    @Override
    public void onPermissionsGranted(int requestCode, List<String> perms) {

    }

    @Override
    public void onPermissionsDenied(int requestCode, List<String> perms) {
        if (EasyPermissions.somePermissionPermanentlyDenied(SplashActivity.this, perms)) {
            openAppDetails();
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == AppSettingsDialog.DEFAULT_SETTINGS_REQ_CODE) {
            setHandler();
        }
    }

    private void openAppDetails() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage(getResources().getString(R.string.app_need_permission));
        builder.setPositiveButton(getResources().getString(R.string.app_permission_by_hand), (dialog, which) -> {
            Intent intent = new Intent();
            intent.setAction(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
            intent.addCategory(Intent.CATEGORY_DEFAULT);
            intent.setData(Uri.parse("package:" + getPackageName()));
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            intent.addFlags(Intent.FLAG_ACTIVITY_NO_HISTORY);
            intent.addFlags(Intent.FLAG_ACTIVITY_EXCLUDE_FROM_RECENTS);
            startActivity(intent);
        });
        builder.setNegativeButton(getResources().getString(R.string.cancel), null);
        builder.show();
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK) {
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    private void setHandler() {
        enterMainView();
    }

    private void enterMainView() {
        Intent intent = new Intent();
        intent.setClass(SplashActivity.this, MainActivity.class);
        startActivity(intent);
        finish();
    }

    private void check() {
        isCheckPrivacy = prefs.getBoolean(Preferences.KEY_PRIVACY, false);
        if (isCheckPrivacy) {
            startPermissionsTask();
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        isCheckPrivacy = prefs.getBoolean(Preferences.KEY_PRIVACY, false);
        if (hasFocus && !isCheckPrivacy){
            showPrivacy();
        }
    }

    private void showPrivacy() {
        mContentView = LayoutInflater.from(SplashActivity.this).inflate(R.layout.popup_user,
                null, false);
        mPopupW = new PopupWindow(mContentView, WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT, false);
        mPopupW.showAtLocation(getWindow().getDecorView(), Gravity.CENTER, 0, 0);
        mPopupW.setContentView(mContentView);
        mPopupW.setTouchable(true);
        mPopupW.setOutsideTouchable(false);
        mPopupW.showAsDropDown(mContentView, 0, 0);
        backgroundAlpha(0.4f);
        mTv_protocol = mContentView.findViewById(R.id.tv_protocol);
        mTv_protocol.setText(getClickableSpan());
        mTv_protocol.setMovementMethod(LinkMovementMethod.getInstance());
        mTv_protocol.setHighlightColor(Color.TRANSPARENT);
        mContentView.findViewById(R.id.pop_agree).setOnClickListener(v -> {
            prefs.edit().putBoolean(Preferences.KEY_PRIVACY, true).apply();
            mPopupW.dismiss();
            startPermissionsTask();
        });
        mContentView.findViewById(R.id.pop_Disagree).setOnClickListener(v -> {
            prefs.edit().putBoolean(Preferences.KEY_PRIVACY, false).apply();
            System.exit(0);
        });
    }

    private void backgroundAlpha(float f) {
        WindowManager.LayoutParams lp = getWindow().getAttributes();
        lp.alpha = f;
        getWindow().setAttributes(lp);
    }

    private CharSequence getClickableSpan() {
        View.OnClickListener l = v -> {
            startActivity(new Intent(SplashActivity.this, PrivacyPolicyActivity.class));
        };
        String User_Agreement = getResources().getString(R.string.me_privacy);
        SpannableString spanableInfo = new SpannableString(User_Agreement);
        String protocol = getResources().getString(R.string.me_user_agreement);
        int start = User_Agreement.indexOf(protocol);
        int end = start + protocol.length();
        spanableInfo.setSpan(new Clickable(l), start, end, Spanned.SPAN_MARK_MARK);
        return spanableInfo;
    }

    class Clickable extends ClickableSpan implements View.OnClickListener {
        private final View.OnClickListener mListener;

        public Clickable(View.OnClickListener l) {
            mListener = l;
        }


        @Override
        public void onClick(View v) {
            mListener.onClick(v);
        }


        @Override
        public void updateDrawState(TextPaint ds) {
            super.updateDrawState(ds);
            ds.setColor(getResources().getColor(R.color.main_tab_text_checked));
            ds.setUnderlineText(false);
        }

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cdvTime != null && cdvTime.isShown()) {
            cdvTime.stop();
        }
        if (mPopupW != null && mPopupW.isShowing()) {
            mPopupW.dismiss();
            mPopupW = null;
        }
    }
}