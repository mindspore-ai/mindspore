/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "kernel/gpu/cuda/other/dynamic_stitch_gpu_kernel.h"
#include <functional>
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/dynamic_stitch_impl.cuh"
#include "kernel/gpu/gpu_common.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kQuad = 4;
size_t GetDtypeNbyte(const std::string &dtype) {
  static const std::unordered_map<std::string, size_t> dtype_nbyte_map = {
    {"float16", sizeof(float) / 2},   {"float32", sizeof(float)},     {"float64", sizeof(float) * 2},
    {"int8", sizeof(int) / kQuad},    {"int16", sizeof(int) / 2},     {"int32", sizeof(int)},
    {"int64", sizeof(int) * 2},       {"uint8", sizeof(int) / kQuad}, {"uint16", sizeof(int) / 2},
    {"uint32", sizeof(int)},          {"uint64", sizeof(int) * 2},    {"bool", sizeof(char)},
    {"complex64", sizeof(float) * 2}, {"bfloat16", sizeof(float) / 2}};

  auto iter = dtype_nbyte_map.find(dtype);
  if (iter != dtype_nbyte_map.end()) {
    return iter->second;
  } else {
    MS_EXCEPTION(ArgumentError) << "Illegal input dtype:" << dtype;
  }
}
}  // namespace

DynamicStitchKernelMod::DynamicStitchKernelMod()
    : n_(0), real_ele_num_(0), max_index_(0), one_data_ele_num_(0), data_type_size_(0) {
  ResetResource();
}

DynamicStitchKernelMod::~DynamicStitchKernelMod() {}

int DynamicStitchKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  // Inputs: (indexlist, datalist)
  size_t input_num = inputs.size();
  n_ = input_num / kDivNum2;

  auto output_shape = GetShapeAdaptively(outputs, 0);
  auto data_type = inputs[n_]->dtype_id();
  // Index type is restricted to int32 by kernel prim.
  size_t index_type_size = sizeof(int);
  data_type_size_ = GetDtypeNbyte(TypeIdToString(data_type, true));
  auto first_data_shape = Convert2SizeTClipNeg(GetShapeAdaptively(inputs, n_));
  auto first_index_dims = GetShapeAdaptively(inputs, 0).size();
  one_data_ele_num_ = 1;
  for (size_t d = first_index_dims; d < first_data_shape.size(); ++d) {
    one_data_ele_num_ *= first_data_shape[d];
  }

  size_t output_size =
    std::accumulate(output_shape.begin(), output_shape.end(), data_type_size_, std::multiplies<size_t>());

  // For max_index
  workspace_size_list_.push_back(index_type_size);
  // One output
  output_size_list_.clear();
  output_size_list_.push_back(output_size);
  return KRET_OK;
}

ShapeVector DynamicStitchKernelMod::GetShapeAdaptively(const std::vector<KernelTensor *> &data, size_t index) {
  auto device_shape = data[index]->GetShapeVector();
  if (IsDynamic(device_shape)) {
    auto ConvertNegOneToDefault = [](int64_t size) {
      constexpr int64_t kDefaultValueForDynamicDim = 16;
      return static_cast<int64_t>(size) < 0 ? kDefaultValueForDynamicDim : size;
    };
    (void)std::transform(device_shape.begin(), device_shape.end(), device_shape.begin(), ConvertNegOneToDefault);
  }

  return device_shape;
}

void DynamicStitchKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "DynamicStitch cudaStreamSynchronized failed");
  auto output_shape = GetShapeAdaptively(outputs, 0);
  output_shape[0] = max_index_ + 1;
  outputs[0]->SetShapeVector(output_shape);
  MS_LOG(DEBUG) << "Run PostExecute for dynamicstitch, real output shape is " << output_shape;
  size_t dtype_byte = GetTypeByte(TypeIdToType(outputs[0]->dtype_id()));
  size_t update_size =
    LongToSize(std::accumulate(output_shape.begin(), output_shape.end(), dtype_byte, std::multiplies<int64_t>()));
  outputs[0]->set_size(update_size);
}

void DynamicStitchKernelMod::ResetResource() noexcept {
  output_size_list_.clear();
  workspace_size_list_.clear();
}

bool DynamicStitchKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  stream_ptr_ = stream;
  auto max_index_dev = GetDeviceAddress<int>(workspace, 0);
  auto output_addr = GetDeviceAddress<unsigned char>(outputs, 0);
  // Init output  and max_index with 0
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(output_addr, 0, output_size_list_[0], cuda_stream),
                                     "Init output with cudamemset failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(max_index_dev, 0, sizeof(int), cuda_stream),
                                     "Init max_index_dev with cudamemset failed");

  for (size_t i = 0; i < n_; i++) {
    auto index_addr = GetDeviceAddress<int>(inputs, i);
    auto data_addr = GetDeviceAddress<unsigned char>(inputs, n_ + i);
    size_t index_num = inputs[i]->size() / sizeof(int);
    auto status = CallStitch(index_addr, data_addr, output_addr, index_num, one_data_ele_num_ * data_type_size_,
                             max_index_dev, cuda_stream);
    CHECK_CUDA_STATUS(status, kernel_name_);
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&max_index_, max_index_dev, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream),
    "For 'DynamicStitch', copy max_index failed");
  if (cudaStreamQuery(cuda_stream) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream), "For 'DynamicStitch', cudaStreamSyncFailed");
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
