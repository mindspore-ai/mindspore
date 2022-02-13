/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_BATCHTOSPACE_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_BATCHTOSPACE_H_
template <typename T>
void CalBatchToSpace(const size_t size, const T *input, const size_t in,
                     const size_t ih, const size_t iw, const size_t ic,
                     const size_t on, const size_t oh, const size_t ow,
                     const size_t oc, const size_t crop_up, const size_t crop_dn,
                     const size_t crop_lft, const size_t crop_rht, const size_t block_num,
                     T *output, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_BATCHTOSPACE_H_
