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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_BATCHNORM_FOLD_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_BATCHNORM_FOLD_H_

template <typename T>
void CalUpdateRunningStd(int channel_size, double epsilon, T* running_std, cudaStream_t cuda_stream);

template <typename T>
void CalUpdateBatchStd(int channel_size, T* batch_std, cudaStream_t cuda_stream);

template <typename T>
void CalBatchNormFoldGrad(const T* d_batch_mean, const T* d_batch_std, const T* x, const T* batch_mean,
                          const T* batch_std, int batch_size, int channel_size, int height, int width, T* dx,
                          cudaStream_t cuda_stream);
template <typename T>
void ThrustFillWith(T* array, int size, T tofill, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_BATCHNORM_FOLD_H_
