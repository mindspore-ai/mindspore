/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SEND_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SEND_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <functional>
#include "plugin/device/gpu/kernel/nccl/nccl_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
class NcclSendGpuKernel : public NcclGpuKernelMod {
 public:
  NcclSendGpuKernel() : dest_rank_(-1) {}
  ~NcclSendGpuKernel() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    (void)Send(input_addr, input_size_list_[0] / sizeof(T), nccl_data_type_, dest_rank_,
               reinterpret_cast<cudaStream_t>(stream_ptr), group_name_);
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 1, but got " << input_num;
    }

    dest_rank_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "dest_rank"));
    group_name_ = GetAttr<std::string>(kernel_node, kAttrGroup);
    nccl_data_type_ = nccl_dtype(AnfAlgo::GetInputDeviceDataType(kernel_node, 0));
    MS_LOG(INFO) << "NcclSend dest rank is " << dest_rank_ << ", group name is " << group_name_;

    auto shape_signed = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    if (IsDynamic(shape_signed)) {
      return true;
    }
    auto input_shape = Convert2SizeTClipNeg(shape_signed);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    size_t input_size = std::accumulate(input_shape.begin(), input_shape.end(), sizeof(T), std::multiplies<size_t>());
    input_size_list_.push_back(input_size);
    output_size_list_.push_back(0);

    SelectCollectiveHandle();
    return true;
  }

 protected:
  void InitSizeLists() override {}

 private:
  int dest_rank_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SEND_GPU_KERNEL_H_
