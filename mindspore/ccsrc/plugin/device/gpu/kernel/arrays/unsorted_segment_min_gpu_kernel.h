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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNSORTED_SEGMENT_MIN_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNSORTED_SEGMENT_MIN_GPU_KERNEL_H_

#include <vector>
#include <limits>
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unsorted_segment_min.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class UnsortedSegmentMinGpuKernelMod : public NativeGpuKernelMod {
 public:
  UnsortedSegmentMinGpuKernelMod() { ResetResource(); }
  ~UnsortedSegmentMinGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    int *indices_addr = GetDeviceAddress<int>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(output_addr, std::numeric_limits<T>::min(), outputs[0]->size,
                                                       reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "cudaMemSet Failed");
    CalUnsortedSegmentMin(input_addr, indices_addr, num_segments_, outer_size_, inner_size_, output_addr,
                          reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
    if (ret != KRET_OK) {
      return ret;
    }
    auto input_shape_signed = inputs[0]->GetShapeVector();
    auto segment_ids_shapes = inputs[1]->GetShapeVector();
    auto output_shapes = outputs[0]->GetShapeVector();
    auto input_shapes = Convert2SizeTClipNeg(input_shape_signed);

    size_t input_num = inputs.size();
    if (input_num == 3) {
      MS_LOG(INFO) << "UnsortedSegmentMin Kernel Input count is 3 - dynamic mode";
    } else {
      MS_LOG(INFO) << "UnsortedSegmentMin Kernel Input count is 2";
    }
    if (output_shapes.size() < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be less than 1, but got "
                        << output_shapes.size();
    }
    num_segments_ = LongToSizeClipNeg(output_shapes[0]);
    input_size_ = SizeOf(input_shape_signed);

    segment_ids_size_ = SizeOf(segment_ids_shapes);
    output_size_ = SizeOf(output_shapes);

    outer_size_ = input_shapes[0];
    inner_size_ = 1;
    for (size_t i = 1; i < input_shapes.size(); i++) {
      inner_size_ *= static_cast<size_t>(input_shapes[i]);
    }

    return KRET_OK;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->name();
    return true;
  }

  void ResetResource() noexcept {
    num_segments_ = 1;
    inner_size_ = 1;
    outer_size_ = 1;
    input_size_ = 1;
    segment_ids_size_ = 1;
    output_size_ = 1;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  int64_t num_segments_;
  size_t inner_size_;
  size_t outer_size_;
  size_t input_size_;
  size_t segment_ids_size_;
  size_t output_size_;
  bool is_null_input_;
  std::string kernel_name_;
  std::vector<int64_t> input_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNSORTED_SEGMENT_MIN_GPU_KERNEL_H_
