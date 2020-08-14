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

#include "nnacl/fp32/roi_pooling.h"
#include <math.h>
#include "nnacl/errorcode.h"

int ROIPooling(float *in_ptr, float *out_ptr, float *roi, const int *in_shape, const int *out_shape, int dim, int tid,
               ROIPoolingParameter *param) {
  int num_rois = out_shape[kNHWC_N];
  int batch_size = in_shape[kNHWC_N];
  int height_ = in_shape[kNHWC_H];
  int width_ = in_shape[kNHWC_W];
  int channels_ = in_shape[kNHWC_C];
  int scale = param->scale_;
  int pooled_height = param->pooledH_;
  int pooled_width = param->pooledW_;
  int in_stride[DIMENSION_4D];
  int out_stride[DIMENSION_4D];
  const int roi_stride = 5;
  in_stride[DIMENSION_4D - 1] = 1;
  out_stride[DIMENSION_4D - 1] = 1;
  for (int i = dim - 2; i >= 0; --i) {
    in_stride[i] = in_stride[i + 1] * in_shape[i + 1];
    out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
  }
  int roi_ind_st = 0;
  for (int i = 0; i < num_rois; ++i) {
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
    float *batch_data = in_ptr + in_stride[kNHWC_N] * roi_batch_ind;

    int out_ind = i * out_stride[0];
    for (int c = kNHWC_N; c < channels_; ++c) {
      float max_v = -__FLT_MAX__;
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int pooled_index =
            i * out_stride[kNHWC_N] + ph * out_stride[kNHWC_H] + pw * out_stride[kNHWC_W] + c * out_stride[kNHWC_C];
          int hstart = (int)floorf(ph * bin_size_h);     // block xi_1
          int wstart = (int)floorf(pw * bin_size_w);     // block yi_1
          int hend = (int)ceilf((ph + 1) * bin_size_h);  // block xi_2
          int wend = (int)ceilf((pw + 1) * bin_size_w);  // block yi_2

          hstart = MSMIN(MSMAX(hstart + roi_start_h, 0), height_);
          hend = MSMIN(MSMAX(hend + roi_start_h, 0), height_);
          wstart = MSMIN(MSMAX(wstart + roi_start_w, 0), width_);
          wend = MSMIN(MSMAX(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);
          if (is_empty) {
            max_v = 0;
          }
          int bd_index = c * in_stride[kNHWC_C] + hstart * in_stride[kNHWC_H];
          for (int h = hstart; h < hend; ++h) {
            int wi = bd_index + wstart * in_stride[kNHWC_W];
            for (int w = wstart; w < wend; ++w) {
              max_v = MSMAX(batch_data[wi], max_v);
              // printf("bd:index: %d, data: %f, max_v: %f\n",wi,batch_data[wi],max_v);
              wi += in_stride[kNHWC_W];
            }
            bd_index += in_stride[kNHWC_H];
          }
          out_ptr[pooled_index] = max_v;
        }
      }
    }
    roi_ind_st += roi_stride;
  }
  return NNACL_OK;
}
