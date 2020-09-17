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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_ROI_ALIGN_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_ROI_ALIGN_IMPL_H_
template <typename T>
void ROIAlign(const T *x, const T *roi_boxes, int roi_rows, int roi_cols, T *out_data, const T spatial_scale,
              const int sample_num, int roi_end_mode, const int channels, const int height, const int width,
              const int pooled_height, const int pooled_width, cudaStream_t cuda_stream);

template <typename T>
void ROIAlignGrad(const T *dy, const T *roi_boxes, int batch_size, int roi_rows, int roi_cols, T *dx,
                  const T spatial_scale, const int sample_num, int roi_end_mode, const int channels, const int height,
                  const int width, const int pooled_height, const int pooled_width, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_ROI_ALIGN_IMPL_H_
