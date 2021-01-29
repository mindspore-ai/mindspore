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
package com.mindspore.posenet;

import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.common.utils.Utils;
import com.mindspore.customview.dialog.NoticeDialog;

@Route(path = "/posenet/PosenetMainActivity")
public class PosenetMainActivity extends AppCompatActivity  {
    private PoseNetFragment poseNetFragment;
    private NoticeDialog noticeDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.posenet_activity_main);
        addCameraFragment();
        init();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_setting_app_posenet, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int itemId = item.getItemId();
        if (itemId == R.id.item_camera) {
            poseNetFragment.switchCamera();
        }else if (itemId == R.id.item_help) {
            showHelpDialog();
        } else if (itemId == R.id.item_more) {
            Utils.openBrowser(this, MSLinkUtils.HELP_POSENET_LITE);
        }
        return super.onOptionsItemSelected(item);
    }


    private void showHelpDialog() {
        noticeDialog = new NoticeDialog(this);
        noticeDialog.setTitleString(getString(R.string.explain_title));
        noticeDialog.setContentString(getString(R.string.explain_posenet));
        noticeDialog.setYesOnclickListener(() -> {
            noticeDialog.dismiss();
        });
        noticeDialog.show();
    }

    private void addCameraFragment() {
        poseNetFragment = PoseNetFragment.newInstance();
        getSupportFragmentManager().popBackStack();
        getSupportFragmentManager().beginTransaction()
                .replace(R.id.container, poseNetFragment)
                .commitAllowingStateLoss();
    }

    private void init() {
        Toolbar mToolbar = findViewById(R.id.posenet_activity_toolbar);
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
    }
}