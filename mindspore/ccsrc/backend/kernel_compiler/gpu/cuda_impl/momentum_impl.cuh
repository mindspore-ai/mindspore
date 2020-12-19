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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_MOMENTUMIMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_MOMENTUMIMPL_H_

#include "runtime/device/gpu/cuda_common.h"
template <typename T, typename S, typename G>
void MomentumUpdateVariable(const size_t size, T *variable, T *accumulation, const S *learning_rate, const G *gradient,
                            const S *momentum, bool use_nesterov, cudaStream_t cuda_stream);
template <typename T, typename S>
void FusedWeightDecayScaleMomentum(const size_t element_num, T *weight_decay, T *scale, T *variable, T *accumulation,
                                   const T *learning_rate, const S *gradient, const T *momentum,
                                   cudaStream_t cuda_stream);
template <typename T, typename S>
void FusedWeightDecayMomentum(const size_t element_num, T *weight_decay, T *variable, T *accumulation,
                              const T *learning_rate, const S *gradient, const T *momentum, cudaStream_t cuda_stream);
template <typename T, typename S>
void FusedScaleMomentum(const size_t element_num, T *scale, T *variable, T *accumulation, const T *learning_rate,
                        const S *gradient, const T *momentum, cudaStream_t cuda_stream);
template <typename T, typename S>
void CombineFusedWeightDecayScaleMomentum(const size_t max, const size_t num, const size_t *element, T **weight_decay,
                                          T **scale, T **variable, T **accumulation, T **learning_rate, S **gradient,
                                          T **momentum, cudaStream_t cuda_stream);
template <typename T, typename S>
void CombineFusedScaleMomentum(const size_t max, const size_t num, const size_t *element, T **scale, T **variable,
                               T **accumulation, T **learning_rate, S **gradient, T **momentum,
                               cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_MOMENTUMIMPL_H_
