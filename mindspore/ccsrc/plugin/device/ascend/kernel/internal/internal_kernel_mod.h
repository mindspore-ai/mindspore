/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_MOD_H_

#include <memory>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <utility>

#include "kernel/kernel.h"
#include "./internal_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/ascend/kernel/internal/tiling_cache.h"
#include "utils/ms_context.h"
#include "include/backend/debug/profiler/profiling.h"

namespace mindspore {
namespace kernel {
static std::map<std::string, int> ms_op_key_to_internel_op_id = {
  {"SiLU", internal::OpId::Swish},
  {"Swiglu", internal::OpId::SwiGLU},
  {"AddLayerNorm", internal::OpId::AddLayerNorm},
  {"Cast", internal::OpId::Cast},
  {"ReshapeAndCache", internal::OpId::ReshapeAndCache},
  {"Gather", internal::OpId::Gather},
  {"ApplyRotaryPosEmb", internal::OpId::ApplyRotaryPosEmb},
  {"Add", internal::OpId::Add},
  {"Sub", internal::OpId::Sub},
  {"RealDiv", internal::OpId::RealDiv},
  {"Mul", internal::OpId::Mul},
  {"Less", internal::OpId::Less},
  {"LogicalNot", internal::OpId::LogicalNot},
  {"NotEqual", internal::OpId::NotEqual},
  {"Equal", internal::OpId::Equal},
  {"Transpose", internal::OpId::Transpose},
  {"GeLU", internal::OpId::Gelu},
  {"Softmax", internal::OpId::Softmax},
  {"RmsNorm", internal::OpId::RmsNorm},
  {"AddRmsNorm", internal::OpId::AddRmsNorm},
  {"ReduceSum", internal::OpId::ReduceSum},
  {"FlashAttentionScore", internal::OpId::FlashAttentionScore},
  {"PagedAttentionMask", internal::OpId::PagedAttention},
  {"PagedAttention", internal::OpId::PagedAttention},
  {"FusedMatMulElemBinary", internal::OpId::MatMul},
  {"FusedMatMulElemUnary", internal::OpId::MatMul},
  {"MatMul", internal::OpId::MatMul},
  {"QuantBatchMatmul", internal::OpId::MatMul},
  {"MatmulSplitOut2", internal::OpId::MatmulQkv},
  {"MatmulSplitOut3", internal::OpId::MatmulQkv},
  {"MatmulBiasSplitOut2", internal::OpId::MatmulQkv},
  {"MatmulBiasSplitOut3", internal::OpId::MatmulQkv},
  {"MatmulBiasSplitSiluOut2", internal::OpId::MatmulQkv},
  {"MatmulSplitSiluOut2", internal::OpId::MatmulQkv},
  {"QuantbatchmatmulSplitOut2", internal::OpId::MatmulQkv},
  {"QuantbatchmatmulSplitOut3", internal::OpId::MatmulQkv},
  {"QuantbatchmatmulSplitSiluOut2", internal::OpId::MatmulQkv},
};

class InternalKernelMod : public KernelMod {
 public:
  explicit InternalKernelMod(std::string &&op_type) : op_type_(std::move(op_type)) {
    ascend_profiler_ = profiler::Profiler::GetInstance(kAscendDevice);
    MS_EXCEPTION_IF_NULL(ascend_profiler_);
  }
  virtual ~InternalKernelMod();

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void set_fullname(const std::string &fullname) { fullname_ = fullname; }

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not support in internal kernel.";
  }

 protected:
  virtual int Build(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  void SetInOutIdx(size_t in_count, size_t out_count);
  virtual internal::OpParamPtr CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) = 0;
  virtual uint64_t GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs);
  virtual void SetTilingInfo(const uint64_t key);
  std::shared_ptr<internal::InternelKernelImpl> impl_;
  std::unordered_map<size_t, size_t> inputsIdxMap_;
  std::unordered_map<size_t, size_t> outputsIdxMap_;
  std::vector<internal::Tensor *> inputs_;
  std::vector<internal::Tensor *> outputs_;
  TilingInfo tiling_info_;
  std::string op_type_;
  std::shared_ptr<profiler::Profiler> ascend_profiler_{nullptr};
  std::string fullname_;
};

using InternalKernelModPtr = std::shared_ptr<InternalKernelMod>;
using InternalKernelModPtrList = std::vector<InternalKernelModPtr>;

#define MS_INTERNAL_KERNEL_FACTORY_REG(NAME, DERIVE) MS_KERNEL_FACTORY_REG(InternalKernelMod, NAME, DERIVE)
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_MOD_H_
