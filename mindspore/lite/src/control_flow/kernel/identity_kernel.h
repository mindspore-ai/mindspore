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

#ifndef MINDSPORE_LITE_SRC_CONTROL_FLOW_KERNEL_IDENTITY_KERNEL_H_
#define MINDSPORE_LITE_SRC_CONTROL_FLOW_KERNEL_IDENTITY_KERNEL_H_
#include <atomic>
#include <utility>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <memory>
#include "src/litert/lite_kernel.h"
#include "src/litert/executor.h"
#include "src/common/log_adapter.h"
#include "src/common/version_manager.h"
#include "src/litert/cpu_info.h"
#include "src/executor/sub_graph_kernel.h"

namespace mindspore::kernel {
// Identity kernel is used to update a reference to a tensor. This is useful in control flow model.
class IdentityKernel : public LiteKernel {
 public:
  IdentityKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(param, inputs, outputs, ctx) {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
    lite::CpuInfo cpu_info;
    support_fp16_ = cpu_info.ArmIsSupportFp16();
#endif
    need_resize_.resize(inputs.size());
  }
  ~IdentityKernel() override = default;
  int PreProcess() override;
  int PostProcess() override;
  int InferShape() override;
  int ReSize() override;
  int Run() override;
  static KernelExec *Create(std::vector<lite::Tensor *> in_tensors, std::vector<lite::Tensor *> out_tensors,
                            const lite::InnerContext *ctx);

 protected:
  int schema_version_ = lite::SCHEMA_VERSION::SCHEMA_CUR;
  std::vector<bool> need_resize_{};
  bool support_fp16_ = false;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_CONTROL_FLOW_KERNEL_IDENTITY_KERNEL_H_
