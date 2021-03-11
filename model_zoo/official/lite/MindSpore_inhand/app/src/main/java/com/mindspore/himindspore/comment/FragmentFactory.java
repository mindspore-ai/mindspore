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
package com.mindspore.himindspore.comment;


import com.mindspore.himindspore.ui.college.CollegeFragment;
import com.mindspore.himindspore.ui.experience.ExperienceFragment;
import com.mindspore.himindspore.ui.experience.VisionFragment;
import com.mindspore.himindspore.ui.me.MeFragment;

public class FragmentFactory {

    private static FragmentFactory mInstance;
    private ExperienceFragment mExperienceFragment;
    private CollegeFragment mCollegeFragment;
    private MeFragment mMeFragment;

    private VisionFragment mVisionFragment;
    private FragmentFactory() {
    }

    public static FragmentFactory getInstance() {
        if (mInstance == null) {
            synchronized (FragmentFactory.class) {
                if (mInstance == null) {
                    mInstance = new FragmentFactory();
                }
            }
        }
        return mInstance;
    }

    public ExperienceFragment getExperienceFragment() {
        if (mExperienceFragment == null) {
            synchronized (FragmentFactory.class) {
                if (mExperienceFragment == null) {
                    mExperienceFragment = new ExperienceFragment();
                }
            }
        }
        return mExperienceFragment;
    }

    public CollegeFragment getCollegeFragment() {
        if (mCollegeFragment == null) {
            synchronized (FragmentFactory.class) {
                if (mCollegeFragment == null) {
                    mCollegeFragment = new CollegeFragment();
                }
            }
        }
        return mCollegeFragment;
    }

    public MeFragment getMeFragment() {
        if (mMeFragment == null) {
            synchronized (FragmentFactory.class) {
                if (mMeFragment == null) {
                    mMeFragment = new MeFragment();
                }
            }
        }
        return mMeFragment;
    }

    public VisionFragment getVisionFragment() {
        if (mVisionFragment == null) {
            synchronized (FragmentFactory.class) {
                if (mVisionFragment == null) {
                    mVisionFragment = new VisionFragment();
                }
            }
        }
        return mVisionFragment;
    }
}
