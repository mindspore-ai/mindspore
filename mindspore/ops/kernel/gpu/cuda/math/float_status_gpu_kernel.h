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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_FLOAT_STATUS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_FLOAT_STATUS_GPU_KERNEL_H_

#include <utility>
#include <algorithm>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/float_status_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/slice_impl.cuh"

namespace mindspore {
namespace kernel {
class FloatStatusGpuKernelMod : public NativeGpuKernelMod {
 public:
  explicit FloatStatusGpuKernelMod(const std::string &kernel_name) { kernel_name_ = kernel_name; }
  ~FloatStatusGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

 private:
  using FloatStatusOpFunc = std::function<bool(FloatStatusGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                               const std::vector<kernel::KernelTensor *> &)>;
  static std::map<std::string, std::vector<std::pair<KernelAttr, FloatStatusGpuKernelMod::FloatStatusOpFunc>>>
    kernel_attr_map_;
  FloatStatusOpFunc kernel_func_;

  enum Optype { OP_STATUS = 0, OP_INF, OP_NAN, OP_FINITE, OP_INVALID = 255 };
  Optype kernel_type_{OP_INVALID};
  size_t input_size_{0};
  size_t output_size_{0};
  size_t type_id_size_{0};
  bool is_null_input_{false};
  void *cuda_stream_{nullptr};

  const std::map<std::string, Optype> kOpTypeMap = {
    {"FloatStatus", OP_STATUS}, {"IsInf", OP_INF}, {"IsNan", OP_NAN}, {"IsFinite", OP_FINITE}};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_FLOAT_STATUS_GPU_KERNEL_H_
