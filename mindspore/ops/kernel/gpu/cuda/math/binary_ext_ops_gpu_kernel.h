/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BINARY_EXT_OPS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BINARY_EXT_OPS_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <algorithm>

#include "kernel/gpu/gpu_kernel.h"
#include "primitive/math_ops.h"
#include "primitive/comparison_ops.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_ops_impl.cuh"
#include "kernel/gpu/kernel_constants.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_types.cuh"

namespace mindspore {
namespace kernel {
constexpr int STRIDE_NUM = 3;
template <typename T>
using Complex = mindspore::utils::Complex<T>;

static const std::map<std::string, BinaryOpType> kBroadcastOpMap = {
  {"AddExt", BinaryOpType::kAddExt},
  {"SubExt", BinaryOpType::kSubExt},
};
class BroadcastExtOptGpuKernelMod : public NativeGpuKernelMod {
 public:
  explicit BroadcastExtOptGpuKernelMod(const std::string &kernel_name) { kernel_name_ = kernel_name; }
  ~BroadcastExtOptGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = reinterpret_cast<cudaStream_t>(cuda_stream);
    return kernel_func_(this, inputs, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <BinaryOpType op, typename In0, typename In1, typename In2, typename OUT>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  using BroadCastFunc = std::function<bool(BroadcastExtOptGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                           const std::vector<kernel::KernelTensor *> &)>;

  BinaryOpType op_type_;
  bool is_broadcast_{false};
  bool is_null_input_{false};
  std::vector<int64_t> simplified_in0_shape_;
  std::vector<int64_t> simplified_in1_shape_;
  std::vector<int64_t> simplified_in2_shape_;
  std::vector<int64_t> simplified_out_shape_;
  cudaStream_t cuda_stream_{nullptr};
  BroadCastFunc kernel_func_{nullptr};
  static std::map<std::string, std::vector<std::pair<KernelAttr, BroadcastExtOptGpuKernelMod::BroadCastFunc>>>
    supported_type_map_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BINARY_EXT_OPS_GPU_KERNEL_H_
