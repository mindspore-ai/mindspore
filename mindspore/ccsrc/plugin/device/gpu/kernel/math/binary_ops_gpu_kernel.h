/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BINARY_OPS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BINARY_OPS_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <algorithm>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/binary_ops_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/binary_types.cuh"

namespace mindspore {
namespace kernel {
constexpr int STRIDE_NUM = 3;
template <typename T>
using Complex = mindspore::utils::Complex<T>;

static const std::map<std::string, BinaryOpType> kBroadcastOpMap = {
  {"Greater", BinaryOpType::kGreater},
  {"Less", BinaryOpType::kLess},
  {"Equal", BinaryOpType::kEqual},
  {"GreaterEqual", BinaryOpType::kGreaterEqual},
  {"LessEqual", BinaryOpType::kLessEqual},
  {"NotEqual", BinaryOpType::kNotEqual},
  {"LogicalAnd", BinaryOpType::kLogicalAnd},
  {"LogicalOr", BinaryOpType::kLogicalOr},
  {"Maximum", BinaryOpType::kMaximum},
  {"Minimum", BinaryOpType::kMinimum},
  {"Mul", BinaryOpType::kMul},
  {"Sub", BinaryOpType::kSub},
  {"Add", BinaryOpType::kAdd},
  {"Div", BinaryOpType::kDiv},
  {"Pow", BinaryOpType::kPow},
  {"RealDiv", BinaryOpType::kRealDiv},
  {"BitwiseAnd", BinaryOpType::kBitwiseAnd},
  {"BitwiseOr", BinaryOpType::kBitwiseOr},
  {"BitwiseXor", BinaryOpType::kBitwiseXor},
  {"Mod", BinaryOpType::kMod},
  {"FloorMod", BinaryOpType::kFloorMod},
  {"SquaredDifference", BinaryOpType::kSquaredDifference},
  {"Atan2", BinaryOpType::kAtan2},
  {"TruncateDiv", BinaryOpType::kTruncateDiv},
  {"TruncateMod", BinaryOpType::kTruncateMod},
  {"AbsGrad", BinaryOpType::kAbsGrad},
  {"FloorDiv", BinaryOpType::kFloorDiv},
  {"DivNoNan", BinaryOpType::kDivNoNan},
  {"MulNoNan", BinaryOpType::kMulNoNan},
  {"Xlogy", BinaryOpType::kXlogy},
  {"Xdivy", BinaryOpType::kXdivy},
  {"Complex", BinaryOpType::kComplex},
};
class BroadcastOptGpuKernelMod : public NativeGpuKernelMod {
 public:
  explicit BroadcastOptGpuKernelMod(const std::string &kernel_name) { kernel_name_ = kernel_name; }
  ~BroadcastOptGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = reinterpret_cast<cudaStream_t>(cuda_stream);
    return kernel_func_(this, inputs, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <BinaryOpType op, typename In0, typename In1, typename OUT>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  using BroadCastFunc = std::function<bool(BroadcastOptGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &)>;

  BinaryOpType op_type_;
  bool is_broadcast_;
  bool is_null_input_;
  std::vector<int64_t> simplified_in0_shape_;
  std::vector<int64_t> simplified_in1_shape_;
  std::vector<int64_t> simplified_out_shape_;
  cudaStream_t cuda_stream_{nullptr};
  BroadCastFunc kernel_func_{nullptr};
  static std::map<std::string, std::vector<std::pair<KernelAttr, BroadcastOptGpuKernelMod::BroadCastFunc>>>
    supported_type_map_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BINARY_OPS_GPU_KERNEL_H_
