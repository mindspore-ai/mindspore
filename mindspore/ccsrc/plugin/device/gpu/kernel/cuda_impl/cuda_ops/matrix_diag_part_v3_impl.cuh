/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MATRIX_DIAG_PART_V3_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MATRIX_DIAG_PART_V3_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT void MatrixDiagPartV3(const T *matrix_ptr, const T *padding_value_ptr, T *diagnal_ptr,
                                      const int64_t num_rows, const int64_t num_cols, const int64_t lower_diag_idx,
                                      const int64_t upper_diag_index, const int64_t diag_size,
                                      const int64_t max_diag_len, const bool left_align_super_diag,
                                      const bool left_align_sub_diag, uint32_t device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MATRIX_DIAG_PART_V3_IMPL_CUH_
