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
#include "backend/kernel_compiler/gpu/other/dynamic_stitch_gpu_kernel.h"
#include <functional>
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/gpu/cuda_impl/dynamic_stitch_impl.cuh"
#include "runtime/device/gpu/gpu_common.h"

namespace mindspore {
namespace kernel {
DynamicStitchKernelMod::DynamicStitchKernelMod()
    : n_(0), real_ele_num_(0), max_index_(0), one_data_ele_num_(0), data_type_size_(0) {
  ResetResource();
}

DynamicStitchKernelMod::~DynamicStitchKernelMod() {}

bool DynamicStitchKernelMod::Init(const CNodePtr &kernel_node) {
  kernel_node_ = kernel_node;
  // Inputs: (indexlist, datalist)
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  n_ = input_num / kDivNum2;

  auto output_shape = AnfAlgo::GetOutputRealDeviceShapeIfExist(kernel_node, 0);
  auto data_type = AnfAlgo::GetInputDeviceDataType(kernel_node, n_);
  // Index type is restricted to int32 by kernel prim.
  size_t index_type_size = sizeof(int);
  data_type_size_ = GetDtypeNbyte(TypeIdToString(data_type, true));
  auto first_data_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, n_);
  one_data_ele_num_ = first_data_shape[first_data_shape.size() - 1];
  for (size_t i = 0; i < n_; i++) {
    auto data_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, n_ + i);
    size_t data_size = std::accumulate(data_shape.begin(), data_shape.end(), 1, std::multiplies<size_t>());
    //  Data size
    input_size_list_.push_back(data_size * data_type_size_);
    // Index size
    input_size_list_.insert(input_size_list_.begin() + i, data_size / one_data_ele_num_ * index_type_size);
  }
  size_t output_size =
    std::accumulate(output_shape.begin(), output_shape.end(), data_type_size_, std::multiplies<size_t>());

  // For max_index
  workspace_size_list_.push_back(index_type_size);
  // One output
  output_size_list_.push_back(output_size);
  return true;
}

void DynamicStitchKernelMod::PostExecute() {
  auto output_shape = AnfAlgo::GetOutputRealDeviceShapeIfExist(kernel_node_.lock(), 0);
  output_shape[0] = max_index_ + 1;
  auto data_type = AnfAlgo::GetInputDeviceDataType(kernel_node_.lock(), n_);
  AnfAlgo::SetOutputInferTypeAndShape({data_type}, {output_shape}, kernel_node_.lock().get());
  MS_LOG(DEBUG) << "Run PostExecute for dynamicstitch, real output shape is " << output_shape;
}

void DynamicStitchKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

void DynamicStitchKernelMod::InitSizeLists() { return; }

bool DynamicStitchKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs, void *stream) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  auto max_index_dev = GetDeviceAddress<int>(workspace, 0);
  auto output_addr = GetDeviceAddress<unsigned char>(outputs, 0);
  // Init output  and max_index with 0
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaMemsetAsync(output_addr, 0, output_size_list_[0], cuda_stream),
                             "Init output with cudamemset failed");
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaMemsetAsync(max_index_dev, 0, sizeof(int), cuda_stream),
                             "Init max_index_dev with cudamemset failed");

  for (size_t i = 0; i < n_; i++) {
    auto index_addr = GetDeviceAddress<int>(inputs, i);
    auto data_addr = GetDeviceAddress<unsigned char>(inputs, n_ + i);
    size_t index_num = input_size_list_[i] / sizeof(int);
    CallStitch(index_addr, data_addr, output_addr, index_num, one_data_ele_num_ * data_type_size_, max_index_dev,
               cuda_stream);
  }
  int temp = 0;
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(&temp, max_index_dev, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream),
                             "Copy max_index failed")
  max_index_ = IntToSize(temp);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
