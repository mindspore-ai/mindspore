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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_GATHER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_GATHER_GPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/framework_ops.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr auto kUnKnown = "UnKnown";
constexpr auto kGather = "Gather";
constexpr auto kSparseGatherV2 = "SparseGatherV2";
class GatherFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  GatherFwdGpuKernelMod() {}
  explicit GatherFwdGpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~GatherFwdGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelAttr> GetOpSupport() override;

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex2}; }

 protected:
  template <typename T, typename S, typename G>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  void Reshape() {
    if (axis_ < 0) {
      axis_ = axis_ + SizeToInt(input_shapes_.size());
    }
    size_t batch_size = 1;
    size_t batch_dims = LongToSize(batch_dims_);
    for (size_t i = 0; i < batch_dims; i++) {
      batch_size *= LongToSize(input_shapes_[i]);
    }
    size_t dim_before_axis = 1;
    for (size_t i = batch_dims; i < std::min(IntToSize(axis_), output_shapes_.size()); i++) {
      dim_before_axis *= LongToSize(output_shapes_[i]);
    }
    size_t dim_of_indices = 1;
    for (size_t i = batch_dims; i < indices_shapes_.size(); i++) {
      dim_of_indices *= LongToSize(indices_shapes_[i]);
    }
    size_t dim_after_indices = 1;
    for (size_t i = IntToSize(axis_) + 1; i < input_shapes_.size(); i++) {
      dim_after_indices *= LongToSize(input_shapes_[i]);
    }
    dims_[kIndex0] = batch_size;
    dims_[kIndex1] = dim_before_axis;
    dims_[kIndex2] = dim_of_indices;
    dims_[kIndex3] = dim_after_indices;
    return;
  }

 private:
  using GatherFunc = std::function<bool(GatherFwdGpuKernelMod *, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, GatherFunc>> func_list_;
  GatherFunc kernel_func_;

  std::vector<int64_t> input_shapes_{};
  std::vector<int64_t> indices_shapes_{};
  std::vector<int64_t> output_shapes_{};
  size_t dims_[kIndex4] = {0};
  int64_t axis_ = 0;
  int64_t batch_dims_{0};
  bool is_null_input_ = false;
  size_t input_type_size_ = 0;
  size_t indices_type_size_ = 0;
  size_t axis_type_size_ = 0;
  std::string kernel_type_{kUnKnown};
  TypeId axis_type_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_GATHER_GPU_KERNEL_H_
