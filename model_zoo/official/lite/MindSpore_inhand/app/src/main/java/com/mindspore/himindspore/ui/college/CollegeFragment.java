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
package com.mindspore.himindspore.ui.college;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.common.sp.Preferences;
import com.mindspore.common.utils.Utils;
import com.mindspore.customview.dialog.NoticeDialog;
import com.mindspore.himindspore.R;
import com.mindspore.himindspore.ui.college.adapter.CollegeItemAdapter;
import com.mindspore.himindspore.ui.college.bean.CollegeItemBean;

import java.util.Arrays;
import java.util.List;

public class CollegeFragment extends Fragment implements CollegeItemAdapter.CollegeItemClickListener {

    private RecyclerView collegeRecycleView;
    private CollegeItemAdapter collegeItemAdapter;

    private List<CollegeItemBean> collegeDataList;
    private SharedPreferences prefs;

    public CollegeFragment() {
        // Required empty public constructor
    }

    public static CollegeFragment newInstance() {
        CollegeFragment fragment;
        fragment = new CollegeFragment();
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_college, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        collegeRecycleView = view.findViewById(R.id.recyclerview);
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        collegeItemAdapter = new CollegeItemAdapter(collegeDataList);
        LinearLayoutManager layoutManager = new LinearLayoutManager(getContext());
        collegeRecycleView.setLayoutManager(layoutManager);
        collegeRecycleView.setAdapter(collegeItemAdapter);
        collegeItemAdapter.setCollegeItemClickListener(this);
    }

    @Override
    public void onAttach(@NonNull Context context) {
        super.onAttach(context);
        prefs = PreferenceManager.getDefaultSharedPreferences(Utils.getApp());
        initData();
    }

    private void initData() {
        collegeDataList = Arrays.asList(
                new CollegeItemBean(CollegeItemBean.TYPE_LEFT_IMAGE_RIGHT_TEXT,
                        R.drawable.college_summary_uncheck, R.drawable.college_summary_checked,
                        getString(R.string.college_summary_title), prefs.getBoolean(Preferences.KEY_COLLEGE_SUMMARY, false)),
                new CollegeItemBean(CollegeItemBean.TYPE_LEFT_IMAGE_RIGHT_TEXT,
                        R.drawable.college_cloud_uncheck, R.drawable.college_cloud_checked,
                        getString(R.string.college_cloud_title), prefs.getBoolean(Preferences.KEY_COLLEGE_CLOUD, false)),
                new CollegeItemBean(CollegeItemBean.TYPE_MIX,
                        R.drawable.college_quick_uncheck, R.drawable.college_quick_checked,
                        getString(R.string.college_quick_title), prefs.getBoolean(Preferences.KEY_COLLEGE_QUICK, false)),
                new CollegeItemBean(CollegeItemBean.TYPE_LEFT_IMAGE_RIGHT_TEXT,
                        R.drawable.college_faq_uncheck, R.drawable.college_faq_checked,
                        getString(R.string.college_faq_title), prefs.getBoolean(Preferences.KEY_COLLEGE_FAQ, false)),
                new CollegeItemBean(CollegeItemBean.TYPE_LEFT_IMAGE_RIGHT_TEXT,
                        R.drawable.college_ask_uncheck, R.drawable.college_ask_checked,
                        getString(R.string.college_ask_title), prefs.getBoolean(Preferences.KEY_COLLEGE_ASK, false)),
                new CollegeItemBean(CollegeItemBean.TYPE_PURE_TEXT,
                        -1, -1, getString(R.string.college_light_title), false));
    }

    private void showSumaryDialog() {
        NoticeDialog noticeDialog = new NoticeDialog(getActivity());
        noticeDialog.setTitleString(getString(R.string.college_dialog_title));
        noticeDialog.setContentString(getString(R.string.college_dialog_content));
        noticeDialog.setGravity(Gravity.START);
        noticeDialog.setShowBottomLogo(true);
        noticeDialog.setYesOnclickListener(() -> {
            noticeDialog.dismiss();
        });
        noticeDialog.show();
    }

    @Override
    public void onCollegeItemClickListener(int position) {
        switch (position) {
            case 0:
                prefs.edit().putBoolean(Preferences.KEY_COLLEGE_SUMMARY, true).apply();
                collegeDataList.get(0).setHasChecked(prefs.getBoolean(Preferences.KEY_COLLEGE_SUMMARY, false));
                collegeItemAdapter.notifyData();
                showSumaryDialog();
                break;
            case 1:
                prefs.edit().putBoolean(Preferences.KEY_COLLEGE_CLOUD, true).apply();
                collegeDataList.get(1).setHasChecked(prefs.getBoolean(Preferences.KEY_COLLEGE_CLOUD, false));
                collegeItemAdapter.notifyData();
                Utils.openBrowser(getActivity(), MSLinkUtils.COLLEGE_MAIN_CLOUD);
                break;
            case 3:
                prefs.edit().putBoolean(Preferences.KEY_COLLEGE_FAQ, true).apply();
                collegeDataList.get(3).setHasChecked(prefs.getBoolean(Preferences.KEY_COLLEGE_FAQ, false));
                collegeItemAdapter.notifyData();
                Utils.openBrowser(getActivity(), MSLinkUtils.COLLEGE_MAIN_FAQ);
                break;
            case 4:
                prefs.edit().putBoolean(Preferences.KEY_COLLEGE_ASK, true).apply();
                collegeDataList.get(4).setHasChecked(prefs.getBoolean(Preferences.KEY_COLLEGE_ASK, false));
                collegeItemAdapter.notifyData();
                Utils.openBrowser(getActivity(), MSLinkUtils.COLLEGE_MAIN_ASK);
                break;
        }
    }

    @Override
    public void onCollegeChildItemClickListener(int position) {
        prefs.edit().putBoolean(Preferences.KEY_COLLEGE_QUICK_ARRAY[position], true).apply();
        boolean isHasCheckedTrain = prefs.getBoolean(Preferences.KEY_COLLEGE_TRAIN, false);
        boolean isHasCheckedExecute = prefs.getBoolean(Preferences.KEY_COLLEGE_EXECUTE, false);
        boolean isHasCheckedApp = prefs.getBoolean(Preferences.KEY_COLLEGE_APP, false);
        boolean isHasCheckedVideo = prefs.getBoolean(Preferences.KEY_COLLEGE_VIDEO, false);
        prefs.edit().putBoolean(Preferences.KEY_COLLEGE_QUICK, isHasCheckedTrain && isHasCheckedExecute && isHasCheckedApp && isHasCheckedVideo).apply();
        collegeDataList.get(2).setHasChecked(prefs.getBoolean(Preferences.KEY_COLLEGE_QUICK, false));
        collegeItemAdapter.notifyData();
        Utils.openBrowser(getActivity(), MSLinkUtils.COLLEGE_QUICK_WEB_ARRAY[position]);
    }
}