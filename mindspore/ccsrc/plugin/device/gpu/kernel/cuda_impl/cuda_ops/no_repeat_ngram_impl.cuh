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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_NO_REPEAT_NGRAM_IMPL_CUH
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_NO_REPEAT_NGRAM_IMPL_CUH
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename StateType, typename LogProbType>
CUDA_LIB_EXPORT void CalculateNoRepeatNGram(const StateType *tokens,
                                            LogProbType *lprobs,
                                            LogProbType *output,
                                            int step,
                                            int no_repeat_ngram_size,
                                            const uint32_t &device_id,
                                            int vocab_size_,
                                            int blocks,
                                            int shared_mem_size,
                                            cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_NO_REPEAT_NGRAM_IMPL_CUH
