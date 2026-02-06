/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARRAY_REDUCE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARRAY_REDUCE_GPU_KERNEL_H_

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/reduce_impl.cuh"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/gpu_kernel_utils.h"
#include "kernel/gpu/kernel_constants.h"
#include "kernel/gpu/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace kernel {
class ArrayReduceGpuKernelMod : public NativeGpuKernelMod {
 public:
  ArrayReduceGpuKernelMod() { ResetResource(); }
  explicit ArrayReduceGpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) { ResetResource(); }
  ~ArrayReduceGpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

 protected:
  void ResetResource() {
    reduce_op_type_ = ReduceSum;
    keep_dims_ = false;
    skip_mode_ = false;
    is_null_input_ = false;
    input_num_ = 1;
    input_size_ = 0;
    kernel_name_ = "ArrayReduce";
    axis_.clear();
    input_reshape_.clear();
    reduce_first_axis_ = false;
    origin_shape_.clear();
    transpose_perm_.clear();
    need_transpose_ = false;
    workspace_size_list_.clear();
  }

  void ResetShapeInfo() {
    input_num_ = 1;
    input_reshape_.clear();
    workspace_size_list_.clear();
    need_transpose_ = false;
    origin_shape_.clear();
    transpose_perm_.clear();
    reduce_first_axis_ = false;
  }

  void InferArrayReduceType();
  void FormatAxis(const size_t &dims, const std::vector<int64_t> &axis, std::vector<bool> *bitmap);
  std::vector<size_t> GetNewShape();
  std::vector<size_t> GetTransposePerm();
  std::vector<size_t> SetOriginalShape();
  void SimplyReduce(const ShapeVector &input_shape, const std::vector<int64_t> &axis);
  TransposeInfo GetTransposeInfo();

  std::vector<KernelAttr> GetOpSupport() override;
  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex1}; }

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  template <typename T>
  bool LaunchComplexKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  template <typename T>
  bool LaunchReduceAllKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
    if (!std::is_same_v<T, bool> && !is_null_input_) {
      auto input_tensor = inputs[kIndex0];
      constexpr size_t kCastWorkspaceIdx = 2;
      auto cast_tensor = workspace[kCastWorkspaceIdx];
      CastKernelTensor<T, bool>(input_tensor, cast_tensor, reinterpret_cast<cudaStream_t>(stream_ptr), kernel_name_);
      auto casted_inputs = inputs;
      casted_inputs[kIndex0] = cast_tensor;
      return LaunchKernel<bool>(casted_inputs, workspace, outputs, stream_ptr);
    }
    return LaunchKernel<bool>(inputs, workspace, outputs, stream_ptr);
  }

  using ReduceFunc = std::function<bool(ArrayReduceGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &, void *)>;
  ReduceFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, ReduceFunc>> all_any_list_;
  static std::vector<std::pair<KernelAttr, ReduceFunc>> prod_list_;
  static std::vector<std::pair<KernelAttr, ReduceFunc>> sum_list_;
  static std::vector<std::pair<KernelAttr, ReduceFunc>> max_min_list_;
  static std::vector<std::pair<KernelAttr, ReduceFunc>> mean_list_;
  static std::map<std::string, std::vector<std::pair<KernelAttr, ReduceFunc>>> kernel_attr_list_;

 private:
  ReduceType_t reduce_op_type_;
  std::vector<int> axis_;
  bool keep_dims_;
  bool skip_mode_;
  bool is_null_input_;
  size_t input_size_;
  std::string kernel_type_{"Unknown"};
  bool reduce_first_axis_;
  std::vector<size_t> input_reshape_;
  size_t input_num_;
  std::vector<size_t> origin_shape_;
  std::vector<size_t> transpose_perm_;
  bool need_transpose_;
  TransposeInfo transpose_info_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARRAY_REDUCE_GPU_KERNEL_H_
