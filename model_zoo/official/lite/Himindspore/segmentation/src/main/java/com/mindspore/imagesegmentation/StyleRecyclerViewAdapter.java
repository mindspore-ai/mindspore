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
package com.mindspore.imagesegmentation;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

public class StyleRecyclerViewAdapter extends RecyclerView.Adapter<StyleRecyclerViewAdapter.StyleItemViewHolder> {

    private final int[] IMAGES;
    private final Context context;
    private final OnBackgroundImageListener mListener;

    public StyleRecyclerViewAdapter(Context context, int[] IMAGES, OnBackgroundImageListener mListener) {
        this.IMAGES = IMAGES;
        this.context = context;
        this.mListener = mListener;
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
//        Glide.with(context).
//                load(IMAGES[position]).
//                into(holder.getImageView());
        holder.imageView.setImageResource(IMAGES[position]);
        holder.itemView.setTag(IMAGES[position]);
        holder.itemView.setOnClickListener(view1 -> {
            if (mListener != null) {
                if (IMAGES.length - 1 == position) {
                    mListener.onImageAdd(holder.itemView);
                } else {
                    mListener.onBackImageSelected(position);
                }
            }
        });
    }


    @Override
    public int getItemCount() {
        return IMAGES == null ? 0 : IMAGES.length;
    }


    public class StyleItemViewHolder extends RecyclerView.ViewHolder {
        private ImageView imageView;
        public StyleItemViewHolder(View mView) {
            super(mView);
            this.imageView = mView.findViewById(R.id.image_view);
        }
    }
}
