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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_BROADCAST_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_BROADCAST_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/broadcast_grad_impl.cuh"
#include "kernel/gpu/kernel_constants.h"
namespace mindspore {
namespace kernel {
template <typename T>
class BroadcastOpGradGpuKernel : public GpuKernel {
 public:
  BroadcastOpGradGpuKernel()
      : op_type_(BROADCAST_GRAD_TYPE_INVALID), need_broadcast_(false), input1_num_(1), input2_num_(1), output_num_(1) {}
  ~BroadcastOpGradGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *x1 = GetDeviceAddress<T>(inputs, 0);
    T *x2 = GetDeviceAddress<T>(inputs, 1);
    T *dy = GetDeviceAddress<T>(inputs, 2);
    T *dx1 = GetDeviceAddress<T>(outputs, 0);
    T *dx2 = GetDeviceAddress<T>(outputs, 1);

    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemsetAsync(dx1, 0, outputs[0]->size, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemSet Failed");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemsetAsync(dx2, 0, outputs[1]->size, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemSet Failed");
    if (need_broadcast_) {
      BroadcastGrad(x1_shape_[0], x1_shape_[1], x1_shape_[2], x1_shape_[3], x2_shape_[0], x2_shape_[1], x2_shape_[2],
                    x2_shape_[3], dy_shape_[0], dy_shape_[1], dy_shape_[2], dy_shape_[3], op_type_, x1, x2, dy, dx1,
                    dx2, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      NoBroadcastGrad(output_num_, op_type_, x1, x2, dy, dx1, dx2, reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    GetOpType(kernel_node);
    auto shape1 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape2 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape3 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    need_broadcast_ = IsBroadcast(shape1, shape2);
    if (need_broadcast_ && shape1.size() > 4) {
      MS_LOG(EXCEPTION) << "Broadcast operation not support dim greater than 4";
    }

    for (size_t i = 0; i < shape3.size(); i++) {
      dy_shape_[i] = shape3[i];
      output_num_ *= shape3[i];
    }
    int offset = shape3.size() - shape1.size();
    for (size_t i = 0; i < shape1.size(); i++) {
      x1_shape_[i + offset] = shape1[i];
      input1_num_ *= shape1[i];
    }
    offset = shape3.size() - shape2.size();
    for (size_t i = 0; i < shape2.size(); i++) {
      x2_shape_[i + offset] = shape2[i];
      input2_num_ *= shape2[i];
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override { return; }
  void InitSizeLists() override {
    input_size_list_.push_back(input1_num_ * sizeof(T));
    input_size_list_.push_back(input2_num_ * sizeof(T));
    input_size_list_.push_back(output_num_ * sizeof(T));
    output_size_list_.push_back(input1_num_ * sizeof(T));
    output_size_list_.push_back(input2_num_ * sizeof(T));
  }

 private:
  void GetOpType(const CNodePtr &kernel_node) {
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);

    static std::map<std::string, BroadcastGradOpType> kBroadcastTypeMap = {
      {"MaximumGrad", BROADCAST_GRAD_TYPE_MAXIMUM},
      {"MinimumGrad", BROADCAST_GRAD_TYPE_MINIMUM},
    };

    auto iter = kBroadcastTypeMap.find(kernel_name);
    if (iter == kBroadcastTypeMap.end()) {
      MS_LOG(EXCEPTION) << "operation " << kernel_name << " is not supported.";
    } else {
      op_type_ = iter->second;
    }
  }

  bool IsBroadcast(const std::vector<size_t> &lhs, const std::vector<size_t> &rhs) {
    if (lhs.size() != rhs.size()) {
      return true;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
      if (lhs[i] != rhs[i]) {
        return true;
      }
    }
    return false;
  }

  BroadcastGradOpType op_type_;
  bool need_broadcast_;
  int input1_num_;
  int input2_num_;
  int output_num_;
  int x1_shape_[4] = {1, 1, 1, 1};
  int x2_shape_[4] = {1, 1, 1, 1};
  int dy_shape_[4] = {1, 1, 1, 1};

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_BINARYOP_GPU_KERNEL_H_
