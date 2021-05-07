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

import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;

import com.mindspore.common.utils.ImageUtils;
import com.mindspore.dance.R;
import com.mindspore.dance.algorithm.ModelDataUtils;
import com.mindspore.dance.global.Variables;


public class ResultFragment extends Fragment {

    private String comment;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_result, container, false);
    }

    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        TextView textFirstView = view.findViewById(R.id.get_score);
        String score = level(Variables.score);
        textFirstView.setText(score);

        TextView textSecondView = view.findViewById(R.id.get_comment);
        if (Variables.score >= 0 && Variables.score <= 100) {
            textSecondView.setText(comment);
        } else {
            textSecondView.setVisibility(View.GONE);
        }
        view.findViewById(R.id.play_again).setOnClickListener(view1 ->
                NavHostFragment.findNavController(ResultFragment.this)
                        .navigate(R.id.action_ResultFragment_to_PrepareFragment));

        view.findViewById(R.id.save).setOnClickListener(view12 -> {
            Uri imgPath = ImageUtils.saveToAlbum(getActivity(), view, null, false);
            if (imgPath != null) {
                Toast.makeText(getActivity(), R.string.image_save_success, Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(getActivity(), R.string.image_save_failed, Toast.LENGTH_SHORT).show();
            }
        });
        view.findViewById(R.id.back).setOnClickListener(view12 -> {
            getActivity().finish();
        });
    }

    private String level(int score) {
        if (score == ModelDataUtils.NO_ACT) {
            return getContext().getString(R.string.get_score_level0);
        } else if (score == ModelDataUtils.NO_POINT) {
            return getContext().getString(R.string.get_score_level8);
        } else if (50 <= score && score < 60) {
            comment = getContext().getString(R.string.get_score_level1);
        } else if (60 <= score && score < 70) {
            comment = getContext().getString(R.string.get_score_level2);
        } else if (70 <= score && score < 80) {
            comment = getContext().getString(R.string.get_score_level3);
        } else if (80 <= score && score < 90) {
            comment = getContext().getString(R.string.get_score_level4);
        } else if (90 <= score && score < 95) {
            comment = getContext().getString(R.string.get_score_level5);
        } else if (95 <= score && score < 100) {
            comment = getContext().getString(R.string.get_score_level6);
        } else if (score == 100) {
            comment = getContext().getString(R.string.get_score_level7);
        } else {
            comment = "";
        }
        return getContext().getString(R.string.get_score) + score;
    }
}
