/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/fp32/roi_pooling_fp32.h"
#include <float.h>
#include <math.h>
#include <string.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"

int ROIPooling(const float *in_ptr, float *out_ptr, const float *roi, float *max_c, int tid,
               const ROIPoolingParameter *param) {
  if (param->thread_num_ == 0) {
    return NNACL_PARAM_INVALID;
  }
  int num_rois = param->output_n_;
  int units = UP_DIV(num_rois, param->thread_num_);
  int roi_st = tid * units;
  int roi_end = MSMIN(num_rois, roi_st + units);
  if (roi_st >= num_rois) {
    return NNACL_OK;
  }
  int batch_size = param->input_n_;
  int height_ = param->input_h_;
  int width_ = param->input_w_;
  int channels_ = param->input_c_;
  float scale = param->scale_;
  int pooled_height = param->pooledH_;
  int pooled_width = param->pooledW_;
  const int roi_stride = 5;
  int roi_ind_st = roi_st * roi_stride;
  for (int i = roi_st; i < roi_end; ++i) {
    int roi_batch_ind = (int)roi[roi_ind_st];  // batch_index
    if (roi_batch_ind >= batch_size) {
      return NNACL_ERRCODE_INDEX_OUT_OF_RANGE;
    }
    int roi_start_h = (int)roundf(roi[roi_ind_st + 1] * scale);  // top-left x1
    int roi_start_w = (int)roundf(roi[roi_ind_st + 2] * scale);  // top-left y1
    int roi_end_h = (int)roundf(roi[roi_ind_st + 3] * scale);    // bottom-right x2
    int roi_end_w = (int)roundf(roi[roi_ind_st + 4] * scale);    // bottom-fight y2

    int roi_height = MSMAX(roi_end_h - roi_start_h + 1, 1);
    int roi_width = MSMAX(roi_end_w - roi_start_w + 1, 1);

    float bin_size_h = (float)roi_height / (float)pooled_height;
    float bin_size_w = (float)roi_width / (float)pooled_width;
    const float *batch_data = in_ptr + param->in_strides_[kNHWC_N] * roi_batch_ind;

    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = (int)floorf(ph * bin_size_h);     // block xi_1
        int wstart = (int)floorf(pw * bin_size_w);     // block yi_1
        int hend = (int)ceilf((ph + 1) * bin_size_h);  // block xi_2
        int wend = (int)ceilf((pw + 1) * bin_size_w);  // block yi_2

        hstart = MSMIN(MSMAX(hstart + roi_start_h, 0), height_);
        hend = MSMIN(MSMAX(hend + roi_start_h, 0), height_);
        wstart = MSMIN(MSMAX(wstart + roi_start_w, 0), width_);
        wend = MSMIN(MSMAX(wend + roi_start_w, 0), width_);
        bool is_empty = (hend <= hstart) || (wend <= wstart);
        for (int j = 0; j < channels_; ++j) {
          max_c[j] = is_empty ? 0 : -FLT_MAX;
        }
        int pooled_index = i * param->out_strides_[0] + ph * param->out_strides_[1] + pw * param->out_strides_[2];
        int bd_index = hstart * param->in_strides_[1];
        for (int h = hstart; h < hend; ++h) {
          int wi = bd_index + wstart * param->in_strides_[2];
          for (int w = wstart; w < wend; ++w) {
            for (int c = 0; c < channels_; ++c) {
              max_c[c] = MSMAX(batch_data[wi + c], max_c[c]);
            }
            wi += param->in_strides_[2];
          }  // in_w end;
          bd_index += param->in_strides_[1];
        }  // in_h end
        for (int j = 0; j < channels_; ++j) {
          out_ptr[pooled_index + j] = max_c[j];
        }
      }
    }
    roi_ind_st += roi_stride;
  }
  return NNACL_OK;
}
