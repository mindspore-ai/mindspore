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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_BRAODCASTTO_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_BRAODCASTTO_GPU_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t SHAPE_SIZE = 4;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
template <typename T, typename S>
class DynamicBroadcastToGpuKernelMod : public NativeGpuKernelMod {
 public:
  DynamicBroadcastToGpuKernelMod() : shape_size_(0), is_null_input_(false) { ResetResource(); }
  ~DynamicBroadcastToGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto data_addr = GetDeviceAddress<T>(inputs, 0);
    auto shape_addr = GetDeviceAddress<S>(inputs, 1);
    auto output_addr = GetDeviceAddress<T>(outputs, 0);

    BroadcastTo(input_shape_[0], input_shape_[1], input_shape_[kIndex2], input_shape_[kIndex3], output_shape_[0],
                output_shape_[1], output_shape_[kIndex2], output_shape_[kIndex3], data_addr, output_addr, cuda_stream);
    real_output_shape_ = std::vector<S>(input_size_list_[1] / sizeof(S), 0);
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(&real_output_shape_[0], shape_addr, input_size_list_[1], cudaMemcpyDeviceToHost, cuda_stream),
      "DynamicBroadcastTo copy real output shape value failed");
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto input_shapes = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    auto shape_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 1);
    auto output_shapes = AnfAlgo::GetOutputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shapes) || CHECK_NULL_INPUT(output_shapes) || CHECK_NULL_INPUT(shape_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'BroadcastToGpuKernelMod', input or output is null";
      InitSizeLists();
      return true;
    }

    if (input_shapes.size() > SHAPE_SIZE || output_shapes.size() > SHAPE_SIZE) {
      MS_LOG(EXCEPTION) << "BroadcastTo operation does not support dim greater than " << SHAPE_SIZE;
    }

    if (output_shapes.size() < input_shapes.size()) {
      MS_LOG(EXCEPTION) << "The rank of BroadcastTo's output [" << output_shapes.size()
                        << "] cannot be smaller than the rank of the input [" << input_shapes.size() << "].";
    }

    shape_size_ = std::accumulate(shape_shape.begin(), shape_shape.end(), sizeof(S), std::multiplies<size_t>());

    size_t offset = output_shapes.size() - input_shapes.size();
    for (size_t i = 0; i < input_shapes.size(); i++) {
      input_shape_[i + offset] = input_shapes[i];
    }

    for (size_t j = 0; j < output_shapes.size(); j++) {
      output_shape_[j] = (output_shapes[j] > 0 ? output_shapes[j] : input_shapes[j]);
    }

    InitSizeLists();
    return true;
  }
  void ResetResource() noexcept override {
    real_output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    for (size_t i = 0; i < SHAPE_SIZE; i++) {
      input_shape_[i] = 1;
      output_shape_[i] = 1;
    }
  }
  void PostExecute() override {
    auto data_type = AnfAlgo::GetInputDeviceDataType(kernel_node_.lock(), 0);
    std::vector<size_t> output_shape;
    std::transform(real_output_shape_.begin(), real_output_shape_.end(), std::back_inserter(output_shape),
                   [](const S &i) { return static_cast<size_t>(i); });
    AnfAlgo::SetOutputInferTypeAndShape({data_type}, {output_shape}, kernel_node_.lock().get());
    MS_LOG(DEBUG) << "Run PostExecute for DynamicBroadcastTo, real output shape is " << output_shape;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_shape_[0] * input_shape_[1] * input_shape_[kIndex2] * input_shape_[kIndex3] *
                               sizeof(T));
    input_size_list_.push_back(shape_size_);
    output_size_list_.push_back(output_shape_[0] * output_shape_[1] * output_shape_[kIndex2] * output_shape_[kIndex3] *
                                sizeof(T));
  }

 private:
  size_t shape_size_;
  size_t input_shape_[SHAPE_SIZE] = {1, 1, 1, 1};
  size_t output_shape_[SHAPE_SIZE] = {1, 1, 1, 1};
  bool is_null_input_ = false;
  std::vector<S> real_output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_BRAODCASTTO_GPU_KERNEL_H_
