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
package com.mindspore.styletransferdemo;

import android.content.Context;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.fragment.app.DialogFragment;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A simple {@link Fragment} subclass.
 * Use the {@link StyleFragment#newInstance} factory method to
 * create an instance of this fragment.
 */
public class StyleFragment extends DialogFragment {

    private OnListFragmentInteractionListener listener;

    public StyleFragment() {
        // Required empty public constructor
    }


    public static StyleFragment newInstance() {
        StyleFragment fragment = new StyleFragment();
        return fragment;
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_style, container, false);
        List<String> styles = new ArrayList<>();
        try {
            styles.addAll(Arrays.asList(getActivity().getAssets().list("thumbnails")));
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (view instanceof RecyclerView) {
            GridLayoutManager gridLayoutManager = new GridLayoutManager(getContext(), 3);
            ((RecyclerView) view).setLayoutManager(gridLayoutManager);
            ((RecyclerView) view).setAdapter(new StyleRecyclerViewAdapter(getActivity(), styles, listener));
        }

        return view;
    }


    @Override
    public void onAttach(@NonNull Context context) {
        super.onAttach(context);
        if (context instanceof OnListFragmentInteractionListener) {
            this.listener = (StyleFragment.OnListFragmentInteractionListener) context;
        }
    }

    public void onDetach() {
        super.onDetach();
        this.listener = null;
    }

    public interface OnListFragmentInteractionListener {
        void onListFragmentInteraction(String item);
    }
}