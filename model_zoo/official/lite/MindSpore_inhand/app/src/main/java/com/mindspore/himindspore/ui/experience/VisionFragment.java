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

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.alibaba.android.arouter.launcher.ARouter;
import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.common.utils.Utils;
import com.mindspore.himindspore.R;

/**
 * A simple {@link Fragment} subclass.
 * Use the {@link VisionFragment#newInstance} factory method to
 * create an instance of this fragment.
 */
@Route(path = "/app/VisionFragment")
public class VisionFragment extends Fragment implements View.OnClickListener {


    public VisionFragment() {
        // Required empty public constructor
    }

    public static VisionFragment newInstance(String param1, String param2) {
        VisionFragment fragment = new VisionFragment();
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
        return inflater.inflate(R.layout.fragment_vision, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        view.findViewById(R.id.btn_object).setOnClickListener(this);  //onClickPhotoDetection
        view.findViewById(R.id.btn_object_camera).setOnClickListener(this);  //onClickCameraDetection
        view.findViewById(R.id.btn_posenet).setOnClickListener(this);  //onClickPoseNet
        view.findViewById(R.id.btn_style_transfer).setOnClickListener(this);  //onClickStyleTransfer
        view.findViewById(R.id.btn_segmentation).setOnClickListener(this);  //onClickSegmentation
        view.findViewById(R.id.btn_image).setOnClickListener(this);  //onClickImage
        view.findViewById(R.id.btn_image_garbage).setOnClickListener(this);  //onClickGarbage
        view.findViewById(R.id.btn_scene).setOnClickListener(this);  //onClickSceneDetection
        view.findViewById(R.id.btn_image_Intelligent_poetry).setOnClickListener(this);  //onClickIntelligentPoetry

    }

    @SuppressLint("NonConstantResourceId")
    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.btn_object:
                ARouter.getInstance().build("/imageobject/ObjectPhotoActivity").navigation();
                break;
            case R.id.btn_object_camera:
                ARouter.getInstance().build("/imageobject/ObjectCameraActivity").navigation();
                break;
            case R.id.btn_posenet:
                ARouter.getInstance().build("/posenet/PosenetMainActivity").navigation();
                break;
            case R.id.btn_style_transfer:
                ARouter.getInstance().build("/styletransfer/StyleMainActivity").navigation();
                break;
            case R.id.btn_segmentation:
                ARouter.getInstance().build("/segmentation/SegmentationMainActivity").navigation();
                break;
            case R.id.btn_image:
                ARouter.getInstance().build("/imageobject/ImageCameraActivity")
                        .withInt("OPEN_TYPE", 1).navigation();
                break;
            case R.id.btn_image_garbage:
                ARouter.getInstance().build("/imageobject/ImageCameraActivity")
                        .withInt("OPEN_TYPE", 2).navigation();
                break;
            case R.id.btn_scene:
                ARouter.getInstance().build("/imageobject/ImageCameraActivity")
                        .withInt("OPEN_TYPE", 3).navigation();
                break;
            case R.id.btn_image_Intelligent_poetry:
                Utils.openBrowser(getActivity(), MSLinkUtils.HELP_INTELLIGENT_POETRY);
                break;

        }
    }
}