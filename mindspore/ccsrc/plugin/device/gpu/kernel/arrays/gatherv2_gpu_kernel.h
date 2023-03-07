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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHERV2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHERV2_GPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gatherv2.cuh"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
constexpr auto kUnKnown = "UnKnown";
constexpr auto kGather = "Gather";
constexpr auto kSparseGatherV2 = "SparseGatherV2";
class GatherV2FwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  GatherV2FwdGpuKernelMod() { ResetResource(); }
  explicit GatherV2FwdGpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) { ResetResource(); }
  ~GatherV2FwdGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  void ResetResource() noexcept {
    input_shapes_.clear();
    indices_shapes_.clear();
    output_shapes_.clear();
    std::fill(dims_, dims_ + 3, 0);
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  template <typename T, typename S, typename G>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  void InitSizeLists() {
    auto input_size = std::accumulate(input_shapes_.begin(), input_shapes_.end(), 1, std::multiplies{});
    auto indices_size = std::accumulate(indices_shapes_.begin(), indices_shapes_.end(), 1, std::multiplies{});
    input_size_list_.push_back(LongToSize(input_size) * input_type_size_);
    input_size_list_.push_back(LongToSize(indices_size) * indices_type_size_);
    auto output_size = std::accumulate(output_shapes_.begin(), output_shapes_.end(), 1, std::multiplies{});
    output_size_list_.push_back(LongToSize(output_size) * input_type_size_);
  }

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
  using GatherV2Func = std::function<bool(GatherV2FwdGpuKernelMod *, const std::vector<AddressPtr> &,
                                          const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, GatherV2Func>> func_list_;
  GatherV2Func kernel_func_;

  std::vector<int64_t> input_shapes_;
  std::vector<int64_t> indices_shapes_;
  std::vector<int64_t> output_shapes_;
  size_t dims_[kIndex4] = {};
  int64_t axis_ = 0;
  int64_t batch_dims_{0};
  bool is_null_input_ = false;
  size_t input_type_size_ = 0;
  size_t indices_type_size_ = 0;
  size_t axis_type_size_ = 0;
  std::string kernel_type_{kUnKnown};
  bool is_get_axis_{false};
  TypeId axis_type_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHERV2_GPU_KERNEL_H_
