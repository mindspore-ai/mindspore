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
package com.mindspore.himindspore.ui.college.adapter;

import android.content.SharedPreferences;
import android.preference.PreferenceManager;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.mindspore.common.sp.Preferences;
import com.mindspore.common.utils.Utils;
import com.mindspore.himindspore.R;
import com.mindspore.himindspore.ui.college.bean.CollegeItemBean;

import java.util.List;

public class CollegeItemAdapter extends RecyclerView.Adapter {

    private List<CollegeItemBean> collegeDataList;
    private CollegeItemClickListener collegeItemClickListener;

    private boolean isHasCheckedTrain;
    private boolean isHasCheckedExecute;
    private boolean isHasCheckedApp;
    private boolean isHasCheckedVideo;

    public CollegeItemAdapter(List<CollegeItemBean> collegeDataList) {
        this.collegeDataList = collegeDataList;
        notifyData();
    }

    public void setCollegeItemClickListener(CollegeItemClickListener collegeItemClickListener) {
        this.collegeItemClickListener = collegeItemClickListener;
    }

    public void notifyData() {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(Utils.getApp());
        this.isHasCheckedTrain = prefs.getBoolean(Preferences.KEY_COLLEGE_TRAIN, false);
        this.isHasCheckedExecute = prefs.getBoolean(Preferences.KEY_COLLEGE_EXECUTE, false);
        this.isHasCheckedApp = prefs.getBoolean(Preferences.KEY_COLLEGE_APP, false);
        this.isHasCheckedVideo = prefs.getBoolean(Preferences.KEY_COLLEGE_VIDEO, false);
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

        View view;
        if (CollegeItemBean.TYPE_LEFT_IMAGE_RIGHT_TEXT == viewType) {
            view = LayoutInflater.from(parent.getContext()).inflate(R.layout.adapter_college_left_image_right_text, parent, false);
            return new LeftImageViewHolder(view);
        } else if (CollegeItemBean.TYPE_MIX == viewType) {
            view = LayoutInflater.from(parent.getContext()).inflate(R.layout.adapter_college_mix, parent, false);
            return new MixViewHolder(view);
        } else if (CollegeItemBean.TYPE_PURE_TEXT == viewType) {
            view = LayoutInflater.from(parent.getContext()).inflate(R.layout.adapter_college_pure_text, parent, false);
            return new PureTextViewHolder(view);
        }
        return null;
    }

    @Override
    public int getItemViewType(int position) {
        return collegeDataList.get(position).getItemType();
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        if (holder instanceof LeftImageViewHolder) {
            bindLeftImageViewHolder((LeftImageViewHolder) holder, position);
        } else if (holder instanceof MixViewHolder) {
            bindMixViewHolder((MixViewHolder) holder, position);
        } else {
            ((PureTextViewHolder) holder).title.setText(collegeDataList.get(position).getTitle());
        }
    }

    private void bindLeftImageViewHolder(LeftImageViewHolder holder, int position) {
        CollegeItemBean bean = collegeDataList.get(position);
        boolean isHasChecked = bean.isHasChecked();
        holder.title.setText(bean.getTitle());

        if (isHasChecked) {
            holder.icon.setImageResource(bean.getImagechecked());
            if (position == collegeDataList.size() - 2) {
                holder.dot.setVisibility(View.GONE);
                holder.lineView.setVisibility(View.GONE);
            } else {
                holder.dot.setVisibility(View.VISIBLE);
                holder.lineView.setVisibility(View.VISIBLE);
                holder.lineView.setBackgroundColor(Utils.getApp().getResources().getColor(R.color.btn_small_checked));
            }
        } else {
            holder.icon.setImageResource(bean.getImageUncheck());
            holder.dot.setVisibility(View.GONE);
            if (position == collegeDataList.size() - 2) {
                holder.lineView.setVisibility(View.GONE);
            } else {
                holder.lineView.setVisibility(View.VISIBLE);
                holder.lineView.setBackgroundColor(Utils.getApp().getResources().getColor(R.color.default_gray));
            }
        }
        holder.itemView.setOnClickListener(view -> {
            if (collegeItemClickListener != null) {
                collegeItemClickListener.onCollegeItemClickListener(position);
            }
        });
    }

    private void bindMixViewHolder(MixViewHolder holder, int position) {
        CollegeItemBean bean = collegeDataList.get(position);
        boolean isHasChecked = bean.isHasChecked();
        holder.title.setText(bean.getTitle());
        if (isHasChecked) {
            holder.icon.setImageResource(bean.getImagechecked());
            holder.dot.setVisibility(View.VISIBLE);
            holder.lineView.setBackgroundColor(Utils.getApp().getResources().getColor(R.color.btn_small_checked));
        } else {
            holder.icon.setImageResource(bean.getImageUncheck());
            holder.dot.setVisibility(View.INVISIBLE);
            holder.lineView.setBackgroundColor(Utils.getApp().getResources().getColor(R.color.default_gray));
        }

        if (isHasCheckedTrain) {
            holder.trainLayout.setBackgroundResource(R.drawable.item_bg_blue_rect);
            holder.iconTrain.setImageResource(R.drawable.college_train_checked);
            holder.textTrain.setTextColor(Utils.getApp().getResources().getColor(R.color.btn_small_checked));
        } else {
            holder.trainLayout.setBackgroundResource(R.drawable.item_bg_gray_rect);
            holder.iconTrain.setImageResource(R.drawable.college_train_uncheck);
            holder.textTrain.setTextColor(Utils.getApp().getResources().getColor(R.color.text_black));
        }
        holder.trainLayout.setOnClickListener(view -> {
            if (collegeItemClickListener != null) {
                collegeItemClickListener.onCollegeChildItemClickListener(0);
            }
        });

        if (isHasCheckedExecute) {
            holder.executeLayout.setBackgroundResource(R.drawable.item_bg_blue_rect);
            holder.iconExecute.setImageResource(R.drawable.college_execute_checked);
            holder.textExecute.setTextColor(Utils.getApp().getResources().getColor(R.color.btn_small_checked));
        } else {
            holder.executeLayout.setBackgroundResource(R.drawable.item_bg_gray_rect);
            holder.iconExecute.setImageResource(R.drawable.college_execute_uncheck);
            holder.textExecute.setTextColor(Utils.getApp().getResources().getColor(R.color.text_black));
        }
        holder.executeLayout.setOnClickListener(view -> {
            if (collegeItemClickListener != null) {
                collegeItemClickListener.onCollegeChildItemClickListener(1);
            }
        });

        if (isHasCheckedApp) {
            holder.appLayout.setBackgroundResource(R.drawable.item_bg_blue_rect);
            holder.iconApp.setImageResource(R.drawable.college_app_checked);
            holder.textApp.setTextColor(Utils.getApp().getResources().getColor(R.color.btn_small_checked));
        } else {
            holder.appLayout.setBackgroundResource(R.drawable.item_bg_gray_rect);
            holder.iconApp.setImageResource(R.drawable.college_app_uncheck);
            holder.textApp.setTextColor(Utils.getApp().getResources().getColor(R.color.text_black));
        }
        holder.appLayout.setOnClickListener(view -> {
            if (collegeItemClickListener != null) {
                collegeItemClickListener.onCollegeChildItemClickListener(2);
            }
        });

        if (isHasCheckedVideo) {
            holder.videoLayout.setBackgroundResource(R.drawable.item_bg_blue_rect);
            holder.iconVideo.setImageResource(R.drawable.college_video_checked);
            holder.textVideo.setTextColor(Utils.getApp().getResources().getColor(R.color.btn_small_checked));
        } else {
            holder.videoLayout.setBackgroundResource(R.drawable.item_bg_gray_rect);
            holder.iconVideo.setImageResource(R.drawable.college_video_uncheck);
            holder.textVideo.setTextColor(Utils.getApp().getResources().getColor(R.color.text_black));
        }
        holder.videoLayout.setOnClickListener(view -> {
            if (collegeItemClickListener != null) {
                collegeItemClickListener.onCollegeChildItemClickListener(3);
            }
        });
    }

    @Override
    public int getItemCount() {
        return collegeDataList.size();
    }


    private class LeftImageViewHolder extends RecyclerView.ViewHolder {

        private ImageView icon;
        private TextView title;
        private ImageView dot;
        private View lineView;

        public LeftImageViewHolder(@NonNull View itemView) {
            super(itemView);
            icon = itemView.findViewById(R.id.icon_left);
            title = itemView.findViewById(R.id.tv_title);
            dot = itemView.findViewById(R.id.bottom_dot);
            lineView = itemView.findViewById(R.id.line_view);
        }
    }

    private class MixViewHolder extends RecyclerView.ViewHolder {

        private ImageView icon;
        private TextView title;
        private ImageView dot;
        private View lineView;

        private RelativeLayout trainLayout;
        private ImageView iconTrain;
        private TextView textTrain;
        private RelativeLayout executeLayout;
        private ImageView iconExecute;
        private TextView textExecute;
        private RelativeLayout appLayout;
        private ImageView iconApp;
        private TextView textApp;
        private RelativeLayout videoLayout;
        private ImageView iconVideo;
        private TextView textVideo;

        public MixViewHolder(@NonNull View itemView) {
            super(itemView);
            icon = itemView.findViewById(R.id.icon_left);
            title = itemView.findViewById(R.id.tv_title);
            dot = itemView.findViewById(R.id.bottom_dot);
            lineView = itemView.findViewById(R.id.line_view);

            trainLayout = itemView.findViewById(R.id.rl_train);
            iconTrain = itemView.findViewById(R.id.icon_train);
            textTrain = itemView.findViewById(R.id.tv_train);

            executeLayout = itemView.findViewById(R.id.rl_execute);
            iconExecute = itemView.findViewById(R.id.icon_execute);
            textExecute = itemView.findViewById(R.id.tv_execute);

            appLayout = itemView.findViewById(R.id.rl_app);
            iconApp = itemView.findViewById(R.id.icon_app);
            textApp = itemView.findViewById(R.id.tv_app);

            videoLayout = itemView.findViewById(R.id.rl_video);
            iconVideo = itemView.findViewById(R.id.icon_video);
            textVideo = itemView.findViewById(R.id.tv_video);
        }
    }

    private class PureTextViewHolder extends RecyclerView.ViewHolder {

        private TextView title;

        public PureTextViewHolder(@NonNull View itemView) {
            super(itemView);
            title = itemView.findViewById(R.id.tv_title);

        }
    }

    @Override
    public long getItemId(int position) {
        return super.getItemId(position);
    }

    public interface CollegeItemClickListener {
        void onCollegeItemClickListener(int position);

        void onCollegeChildItemClickListener(int position);

    }


}
