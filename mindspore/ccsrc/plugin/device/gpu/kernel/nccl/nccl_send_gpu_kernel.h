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

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    (void)Send(input_addr, input_size_list_[0] / sizeof(T), nccl_data_type_, dest_rank_,
               reinterpret_cast<cudaStream_t>(stream_ptr), group_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    size_t input_num = inputs.size();
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
    }
    SelectCollectiveHandle();
    return true;
  }
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    dest_rank_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("dest_rank")));
    group_name_ = GetValue<std::string>(primitive_->GetAttr(kAttrGroup));
    nccl_data_type_ = nccl_dtype(inputs[0]->dtype_id());
    MS_LOG(INFO) << "NcclSend dest rank is " << dest_rank_ << ", group name is " << group_name_;

    auto shape_signed = outputs[0]->GetDeviceShapeVector();
    if (IsDynamic(shape_signed)) {
      return KRET_UNKNOWN_OUT_SHAPE;
    }
    auto input_shape = Convert2SizeTClipNeg(shape_signed);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      return true;
    }
    output_size_list_.clear();
    output_size_list_.push_back(0);
    return KRET_OK;
  }

 private:
  int dest_rank_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SEND_GPU_KERNEL_H_
