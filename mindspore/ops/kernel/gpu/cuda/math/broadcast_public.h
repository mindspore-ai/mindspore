/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_BINARY_BROADCAST_PUB_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_BINARY_BROADCAST_PUB_H_

#include <stdint.h>
#include <vector>
#include "kernel/gpu/gpu_kernel.h"
bool IsBinaryBroadcast(const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape);
void CalSimplifyShape(const std::vector<int64_t> &aligned_in0_shape, const std::vector<int64_t> &aligned_in1_shape,
                      const std::vector<int64_t> &aligned_out_shape, std::vector<int64_t> *simplified_in0_shape,
                      std::vector<int64_t> *simplified_in1_shape, std::vector<int64_t> *simplified_out_shape);
void SimplifyBinaryBroadcastShape(const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,
                                  const std::vector<int64_t> &out_shape, std::vector<int64_t> *simplified_in0_shape,
                                  std::vector<int64_t> *simplified_in1_shape,
                                  std::vector<int64_t> *simplified_out_shape);
void SimplifyBroadcastToShape(const std::vector<int64_t> &inp_shape, const std::vector<int64_t> &out_shape,
                              std::vector<int64_t> *simplified_inp_shape, std::vector<int64_t> *simplified_out_shape);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_BINARY_BROADCAST_PUB_H_
