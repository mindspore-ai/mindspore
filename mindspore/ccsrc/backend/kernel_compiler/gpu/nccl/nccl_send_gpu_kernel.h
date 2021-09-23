/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/gpu/nccl/nccl_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
class NcclSendGpuKernel : public NcclGpuKernel {
 public:
  NcclSendGpuKernel() : dest_rank_(-1), collective_handle_(nullptr) {}
  ~NcclSendGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    auto nccl_send_func = reinterpret_cast<Send>(dlsym(const_cast<void *>(collective_handle_), "Send"));
    MS_EXCEPTION_IF_NULL(nccl_send_func);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_,
                               (*nccl_send_func)(input_addr, input_size_list_[0] / sizeof(T), nccl_data_type_,
                                                 dest_rank_, reinterpret_cast<cudaStream_t>(stream_ptr), group_name_),
                               "ncclSend failed");
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but NCCL send needs 1 input.";
      return false;
    }

    dest_rank_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "dest_rank"));
    group_name_ = GetAttr<std::string>(kernel_node, kAttrGroup);
    nccl_data_type_ = nccl_dtype(AnfAlgo::GetInputDeviceDataType(kernel_node, 0));
    MS_LOG(INFO) << "NcclSend dest rank is " << dest_rank_ << ", group name is " << group_name_;

    auto input_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'NcclSendGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    size_t input_size = std::accumulate(input_shape.begin(), input_shape.end(), sizeof(T), std::multiplies<size_t>());
    input_size_list_.push_back(input_size);
    output_size_list_.push_back(0);

    collective_handle_ = device::gpu::CollectiveInitializer::instance().collective_handle();
    MS_EXCEPTION_IF_NULL(collective_handle_);
    return true;
  }

 protected:
  void InitSizeLists() override {}

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int dest_rank_;
  bool is_null_input_;
  const void *collective_handle_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SEND_GPU_KERNEL_H_
