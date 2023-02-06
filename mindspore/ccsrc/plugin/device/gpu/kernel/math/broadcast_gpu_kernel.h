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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BROADCAST_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BROADCAST_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <algorithm>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/binary_func.cuh"

namespace mindspore {
namespace kernel {
constexpr int MAX_DIMS = 7;

static const std::map<std::string, BinaryOpType> kBroadcastCmpTypeMap = {
  {"Greater", BinaryOpType::kGreater},       {"Less", BinaryOpType::kLess},
  {"Equal", BinaryOpType::kEqual},           {"GreaterEqual", BinaryOpType::kGreaterEqual},
  {"LessEqual", BinaryOpType::kLessEqual},   {"NotEqual", BinaryOpType::kNotEqual},
  {"LogicalAnd", BinaryOpType::kLogicalAnd}, {"LogicalOr", BinaryOpType::kLogicalOr},
};

static const std::map<std::string, BinaryOpType> kBroadcastArithmetricTypeMap = {
  {"Maximum", BinaryOpType::kMaximum},
  {"Minimum", BinaryOpType::kMinimum},
  {"Pow", BinaryOpType::kPower},
  {"RealDiv", BinaryOpType::kRealDiv},
  {"Mul", BinaryOpType::kMul},
  {"Sub", BinaryOpType::kSub},
  {"Add", BinaryOpType::kAdd},
  {"FloorDiv", BinaryOpType::kFloorDiv},
  {"AbsGrad", BinaryOpType::kAbsGrad},
  {"Div", BinaryOpType::kDiv},
  {"DivNoNan", BinaryOpType::kDivNoNan},
  {"MulNoNan", BinaryOpType::kMulNoNan},
  {"Mod", BinaryOpType::kMod},
  {"FloorMod", BinaryOpType::kFloorMod},
  {"Atan2", BinaryOpType::kAtan2},
  {"TruncateDiv", BinaryOpType::kTruncateDiv},
  {"TruncateMod", BinaryOpType::kTruncateMod},
  {"BitwiseAnd", BinaryOpType::kBitwiseAnd},
  {"BitwiseOr", BinaryOpType::kBitwiseOr},
  {"BitwiseXor", BinaryOpType::kBitwiseXor},
  {"Xdivy", BinaryOpType::kXdivy},
  {"Xlogy", BinaryOpType::kXlogy},
};

static const std::map<std::string, BinaryOpType> kBroadcastComplexAndRealTypeMap = {
  {"RealDiv", BinaryOpType::kRealDiv}, {"Mul", BinaryOpType::kMul},     {"Sub", BinaryOpType::kSub},
  {"Add", BinaryOpType::kAdd},         {"Div", BinaryOpType::kDiv},     {"MulNoNan", BinaryOpType::kMulNoNan},
  {"Pow", BinaryOpType::kPower},       {"Xdivy", BinaryOpType::kXdivy}, {"Xlogy", BinaryOpType::kXlogy}};

static const std::map<std::string, BinaryOpType> kBroadcastComplexOnlyTypeMap = {
  {"Complex", BinaryOpType::kComplex},
};

class BroadcastOpGpuKernelMod : public NativeGpuKernelMod {
 public:
  BroadcastOpGpuKernelMod() {}
  ~BroadcastOpGpuKernelMod() override = default;

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
  bool GetOpType();

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename T, typename S, typename G>
  bool LaunchComplexKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  using BroadCastFunc = std::function<bool(BroadcastOpGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &)>;

  std::string GetValidKernelTypes();

  BinaryOpType op_type_;
  bool need_broadcast_;
  bool is_compare_op_;
  bool support_complex_{true};
  bool support_real_{true};
  bool is_null_input_;
  std::vector<size_t> lhs_shape_;
  std::vector<size_t> rhs_shape_;
  std::vector<size_t> output_shape_;
  size_t unit_size_{1};
  size_t output_num_{1};
  cudaStream_t cuda_stream_{nullptr};
  BroadCastFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, BroadCastFunc>> real_list_;
  static std::vector<std::pair<KernelAttr, BroadCastFunc>> complex_list_;
  std::vector<std::pair<KernelAttr, BroadcastOpGpuKernelMod::BroadCastFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BROADCAST_GPU_KERNEL_H_
