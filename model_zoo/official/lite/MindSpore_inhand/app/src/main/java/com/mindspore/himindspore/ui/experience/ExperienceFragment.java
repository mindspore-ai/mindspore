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
package com.mindspore.himindspore.ui.experience;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.viewpager.widget.ViewPager;

import com.mindspore.common.base.adapter.BasePagerAdapter;
import com.mindspore.himindspore.R;
import com.mindspore.himindspore.comment.FragmentFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class ExperienceFragment extends Fragment {

    private ViewPager vpContent;

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_experience, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        vpContent = view.findViewById(R.id.vp_content);
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        initData();

    }

    public void initData() {
        List<Fragment> fragmentList = new ArrayList<>();
        String[] categoryName = this.getResources().getStringArray(R.array.tab_experience);
        fragmentList.add(FragmentFactory.getInstance().getVisionFragment());

        BasePagerAdapter adapter = new BasePagerAdapter(getChildFragmentManager(), fragmentList, Arrays.asList(categoryName));
        vpContent.setAdapter(adapter);
        vpContent.setOffscreenPageLimit(categoryName.length);
    }
}