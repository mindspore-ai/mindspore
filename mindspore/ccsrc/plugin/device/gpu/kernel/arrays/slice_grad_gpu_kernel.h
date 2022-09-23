/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_

#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include <map>
#include <memory>
#include "mindspore/core/ops/op_name.h"
#include "mindspore/core/ops/grad/slice_grad.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/slice_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/slice_grad_helper.h"

namespace mindspore {
namespace kernel {
constexpr size_t kSliceGradDefaultInputShapeSize = 4;
constexpr size_t kSliceGradMaxInputShapeSize = 7;
constexpr size_t DynamicInputNum = 4;
constexpr size_t kBeginIndex_ = 2;
constexpr size_t kSizeIndex_ = 3;
constexpr size_t kDim4 = 4;
constexpr size_t kDim7 = 7;

class SliceGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  SliceGradGpuKernelMod();
  ~SliceGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ProccessAttr(const std::vector<KernelTensorPtr> &inputs);
  void CalcBeginAndSize(const mindspore::Format &data_format, size_t dim = kDim4);
  void CheckParam(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs);

  std::vector<int64_t> begin_;
  std::vector<int64_t> size_;
  std::vector<int64_t> strides_;
  ShapeVector input_shape_;
  ShapeVector dy_shape_;

  bool is_dynamic_attr_{false};
  bool get_dynamic_attr_value_{false};
  std::string kernel_name_;
  std::shared_ptr<cukernel::SliceGradAttr> attr_ptr_{nullptr};

  std::unique_ptr<cukernel::GpuKernelHelperBase> helper_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_
