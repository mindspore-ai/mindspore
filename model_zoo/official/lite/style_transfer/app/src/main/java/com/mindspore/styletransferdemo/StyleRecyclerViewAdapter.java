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
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;

import java.util.List;

public class StyleRecyclerViewAdapter extends RecyclerView.Adapter<StyleRecyclerViewAdapter.StyleItemViewHolder> {

    private View.OnClickListener mOnClickListener;
    private List<String> stylesList;
    private Context context;
    private StyleFragment.OnListFragmentInteractionListener mListener;

    public StyleRecyclerViewAdapter(Context context, List<String> stylesList, StyleFragment.OnListFragmentInteractionListener mListener) {
        this.stylesList = stylesList;
        this.context = context;
        this.mListener = mListener;

        this.mOnClickListener = new View.OnClickListener() {
            @Override
            public void onClick(View view) {

            }
        };

        this.mOnClickListener = (View.OnClickListener) (new View.OnClickListener() {
            public final void onClick(View v) {

                if (v.getTag() != null && v.getTag() instanceof String) {
                    mListener.onListFragmentInteraction(String.valueOf(v.getTag()));
                }
            }
        });
    }

    @NonNull
    @Override
    public StyleItemViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context)
                .inflate(R.layout.image_item, parent, false);
        return new StyleItemViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull StyleItemViewHolder holder, int position) {
        String imagePath = stylesList.get(position);
        Glide.with(context).
                load(Uri.parse("file:///android_asset/thumbnails/" + imagePath)).
                centerInside().
                into(holder.getImageView());

        View view = holder.getMView();
        view.setTag(imagePath);
        view.setOnClickListener(this.mOnClickListener);
    }


    @Override
    public int getItemCount() {
        return stylesList == null ? 0 : stylesList.size();
    }


    public class StyleItemViewHolder extends RecyclerView.ViewHolder {
        private ImageView imageView;
        private final View mView;

        public final ImageView getImageView() {
            return this.imageView;
        }

        public final void setImageView(ImageView imageView) {
            this.imageView = imageView;
        }

        public final View getMView() {
            return this.mView;
        }

        public StyleItemViewHolder(View mView) {
            super(mView);
            this.mView = mView;
            this.imageView = mView.findViewById(R.id.image_view);
        }
    }
}
