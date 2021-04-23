package com.mindspore.hms.ImageSegmentation;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.mindspore.hms.R;

public class PortraitSegmentationAdapter extends RecyclerView.Adapter<PortraitSegmentationAdapter.PortraitItemViewHolder> {


    private final int[] IMAGES;
    private final Context context;
    private final OnBackgroundImageListener mListener;

    public PortraitSegmentationAdapter(Context context, int[] IMAGES, OnBackgroundImageListener mListener) {
        this.IMAGES = IMAGES;
        this.context = context;
        this.mListener = mListener;
    }

    @NonNull
    @Override
    public PortraitItemViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context)
                .inflate(R.layout.image_item, parent, false);
        return new PortraitItemViewHolder(view);
    }


    @Override
    public void onBindViewHolder(@NonNull PortraitItemViewHolder holder, int position) {
        Glide.with(context).
                load(IMAGES[position]).
                into(holder.getImageView());

        View view = holder.getMView();
        view.setTag(IMAGES[position]);
        view.setOnClickListener(view1 -> {
            if (mListener != null) {
                if (IMAGES.length - 1 == position) {
                    mListener.onImageAdd(holder.getImageView());
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


    public class PortraitItemViewHolder extends RecyclerView.ViewHolder {
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

        public PortraitItemViewHolder(View mView) {
            super(mView);
            this.mView = mView;
            this.imageView = mView.findViewById(R.id.image_view);
        }
    }
}
