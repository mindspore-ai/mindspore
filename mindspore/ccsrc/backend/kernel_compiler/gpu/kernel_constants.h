/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_KERNEL_CONSTANTS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_KERNEL_CONSTANTS_H_
#include <map>
#include <string>

namespace mindspore {
namespace kernel {
// Used by Pooling and Conv2d
static constexpr char kSamePadModeUpperCase[] = "SAME";

// Used by Pooling and Conv2d
static constexpr char kSamePadModeLowerCase[] = "same";

// Used by Pooling and Conv2d
static constexpr char kValidPadModeUpperCase[] = "VALID";

// Used by Pooling and Conv2d
static constexpr char kValidPadModeLowerCase[] = "valid";

// Used by Pooling
static constexpr char kAvgPoolingModeUpperCase[] = "AVG";

// Used by Pooling
static constexpr char kAvgPoolingModeLowerCase[] = "avg";

// Used by MaxPool pad: The minimum value of float32
static constexpr float kSignedMinFloat = -3.402823466e+38F;

// Used by mixprecision, cudnn dtype select
static std::map<std::string, cudnnDataType_t> kCudnnDtypeMap = {
  {"kNumberTypeFloat32", CUDNN_DATA_FLOAT},  {"kNumberTypeFloat16", CUDNN_DATA_HALF},
  {"kNumberTypeFloat64", CUDNN_DATA_DOUBLE}, {"kNumberTypeInt32", CUDNN_DATA_INT32},
  {"kNumberTypeBool", CUDNN_DATA_INT8},      {"kNumberTypeInt8", CUDNN_DATA_INT8},
  {"kNumberTypeUInt8", CUDNN_DATA_UINT8}};
// Used by mixprecision, cuda dtype select
static std::map<std::string, cudaDataType_t> kCudaDtypeMap = {
  {"kNumberTypeFloat64", CUDA_R_64F}, {"kNumberTypeFloat32", CUDA_R_32F}, {"kNumberTypeFloat16", CUDA_R_16F}};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_KERNEL_CONSTANTS_H_
