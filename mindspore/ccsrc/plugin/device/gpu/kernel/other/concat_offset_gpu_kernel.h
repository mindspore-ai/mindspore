/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_CONCAT_OFFSET_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_CONCAT_OFFSET_GPU_KERNEL_H_

#include <map>
#include <vector>
#include <string>
#include "mindspore/core/ops/concat_offset.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class ConcatOffsetGpuKernelMod : public NativeGpuKernelMod {
 public:
  ConcatOffsetGpuKernelMod() {}
  ~ConcatOffsetGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    S *output_device_address = GetDeviceAddress<S>(outputs, 0);
    size_t out_size = out_offset_.size() * sizeof(S);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_device_address, out_offset_.data(), out_size, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync error in ConcatOffsetGpuKernelMod::Launch");
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) {
    kernel_name_ = base_operator->GetPrim()->name();
    constexpr size_t outputs_num = 1;
    if (outputs.size() != outputs_num) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << outputs.size();
      return false;
    }
    if (inputs.size() == 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of input is 0";
      return false;
    }
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
    ResetResource();
    if (inputs[kIndex0]->IsDynamicShape()) {
      return KRET_UNKNOWN_SHAPE;
    }
    auto input_shape = inputs[kIndex0]->GetShapeVector();
    auto rank = input_shape.size();
    auto rank_int = SizeToInt(rank);
    auto kernel_ptr = std::dynamic_pointer_cast<ops::ConcatOffset>(base_operator);
    int64_t axis = 0;
    if (kernel_ptr->HasAttr(kAttrAxis)) {
      axis = kernel_ptr->get_axis();
    }
    if (axis < -rank_int || axis >= rank_int) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' should be in the range [-" << rank << "," << rank
                        << "), but got " << axis;
    }
    if (axis < 0) {
      axis += rank_int;
    }
    size_t input_num = inputs.size();
    for (size_t i = 0; i < input_num; i++) {
      int64_t input_size = 1;
      auto input_shape_i = inputs[i]->GetDeviceShapeAdaptively();
      for (size_t j = 0; j < input_shape_i.size(); j++) {
        input_size *= input_shape_i[j];
      }
      input_size_list_.push_back(LongToSizeClipNeg(input_size) * sizeof(T));
    }
    // cal offset
    int64_t shape_offset = input_shape[axis];
    std::vector<size_t> offset(input_num, 0);
    for (size_t i = 1; i < input_num; i++) {
      input_shape = inputs[i]->GetShapeVector();
      if (input_shape.size() != rank) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the dimension of input should be equal, but got:"
                          << " the dimension of the " << i << "'th input: " << input_shape.size()
                          << " and the dimension of the first input:  " << rank;
      }
      offset[i] = LongToSizeClipNeg(shape_offset);
      shape_offset += input_shape[axis];
    }
    constexpr size_t kConcatOffsetOutputShapeSize = 2;
    auto output_shape = outputs[0]->GetShapeVector();
    if (output_shape.size() != kConcatOffsetOutputShapeSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output should be "
                        << kConcatOffsetOutputShapeSize << ", but got:" << output_shape.size();
    }
    if (output_shape[0] != SizeToInt(input_num)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the first dimension value of output should be equal to "
                           "the number of input, but got the first dimension value of output: "
                        << output_shape[0] << ", and the number of input: " << input_num;
    }
    if (output_shape[1] != rank_int) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the second dimension value of output should be equal to "
                           "the dimension of input, but got the second dimension value of output: "
                        << output_shape[1] << ", and the dimension of input: " << rank;
    }
    auto output_size = input_num * rank;
    out_offset_.assign(output_size, 0);
    for (size_t i = 0; i < input_num; ++i) {
      out_offset_[i * rank + axis] = offset[i];
    }
    output_size_list_.push_back(out_offset_.size() * sizeof(S));
    return KRET_OK;
  }

  void ResetResource() {
    input_size_list_.clear();
    output_size_list_.clear();
    out_offset_.clear();
  }

 private:
  std::vector<S> out_offset_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_OTHER_CONCAT_OFFSET_GPU_KERNEL_H_
