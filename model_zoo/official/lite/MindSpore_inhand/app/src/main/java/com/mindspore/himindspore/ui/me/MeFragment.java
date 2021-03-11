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
package com.mindspore.himindspore.ui.me;

import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.common.utils.Utils;
import com.mindspore.himindspore.R;
import com.mindspore.himindspore.ui.main.PrivacyPolicyActivity;

public class MeFragment extends Fragment implements View.OnClickListener {

    private final String TAG = "MeFragment";
    private TextView versionText;

    @Override
    public View onCreateView(
            @NonNull LayoutInflater inflater, @Nullable ViewGroup container,
            @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_me, container, false);
        return view;
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        versionText = view.findViewById(R.id.me_vision);
        view.findViewById(R.id.rl_me_share).setOnClickListener(this);
        view.findViewById(R.id.rl_me_thumbsup).setOnClickListener(this);
        view.findViewById(R.id.rl_me_official).setOnClickListener(this);
        view.findViewById(R.id.rl_me_official_code).setOnClickListener(this);
        view.findViewById(R.id.rl_me_qa).setOnClickListener(this);
        view.findViewById(R.id.rl_me_version).setOnClickListener(this);
        view.findViewById(R.id.me_user_protocol).setOnClickListener(this);
        showPackageInfo();
    }

    public void onClickShare() {
        Intent share_intent = new Intent();
        share_intent.setAction(Intent.ACTION_SEND);
        share_intent.setType("text/plain");
        share_intent.putExtra(Intent.EXTRA_SUBJECT, getString(R.string.title_share));
        share_intent.putExtra(Intent.EXTRA_TEXT, getString(R.string.title_share_commend) + MSLinkUtils.ME_APK_URL);
        share_intent = Intent.createChooser(share_intent, getString(R.string.title_share));
        startActivity(share_intent);
    }

    private void showPackageInfo() {
        try {
            PackageManager packageManager = this.getActivity().getPackageManager();
            PackageInfo packageInfo = packageManager
                    .getPackageInfo(this.getActivity().getPackageName(), 0);
            versionText.setText("V" + packageInfo.versionName);
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.rl_me_share:
                onClickShare();
                break;
            case R.id.rl_me_thumbsup:
                Utils.openBrowser(getActivity(), MSLinkUtils.ME_STAR_URL);
                break;
            case R.id.rl_me_official:
                Utils.openBrowser(getActivity(), MSLinkUtils.BASE_URL);
                break;
            case R.id.rl_me_official_code:
                Utils.openBrowser(getActivity(), MSLinkUtils.ME_CODE_URL);
                break;
            case R.id.rl_me_qa:
                Utils.openBrowser(getActivity(), MSLinkUtils.ME_HELP_URL);
                break;
            case R.id.me_user_protocol:
                startActivity(new Intent(getContext(), PrivacyPolicyActivity.class));
                break;
        }
    }
}
