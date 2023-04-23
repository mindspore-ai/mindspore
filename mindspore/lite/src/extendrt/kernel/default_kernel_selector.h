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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_DEFAULT_KERNEL_SELECTOR_H
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_DEFAULT_KERNEL_SELECTOR_H

#include <vector>
#include <map>
#include <string>
#include "src/extendrt/kernel/kernel_selector.h"
#include "src/extendrt/kernel/nnacl/nnacl_lib.h"
#include "src/extendrt/kernel/default/default_kernel_lib.h"

namespace mindspore::kernel {
class DefaultKernelSelector : public KernelSelector {
 public:
  explicit DefaultKernelSelector(const infer::abstract::CompileOption *compile_option)
      : KernelSelector(compile_option) {}
  ~DefaultKernelSelector() override = default;

  LiteKernel *CreateKernel(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                           const std::vector<InferTensor *> &outputs, const InferContext *ctx) override {
    static std::map<std::string, int> kPriorityMap{
      {"custom", 0},
      {kNNAclName, 1},
      {"bolt", 1},
      {kDefaultKernelLibName, 100},
    };
    // Optimize:
    // some operators have a format preference while others do not.
    // 1. if operator has a format preference, select kernel by format firstly and by priority secondly
    // 2. if not, select only by priority
    // but now, we just ignore whether the operator has a preference or not, just select kernel by format firstly and
    // by priority secondly
    auto candidates = Candidates(spec.op_type, spec.attr, true);
    if (candidates.empty()) {
      candidates = Candidates(spec.op_type, spec.attr);
    }
    if (candidates.empty()) {
      MS_LOG(ERROR) << "Can not find suitable kernellib, op_type: " << spec.op_type << ", kernel attr: " << spec.attr;
      return nullptr;
    }
    int min_priority = INT32_MAX;
    const kernel::KernelLib *selected{nullptr};
    for (auto &candidate : candidates) {
      auto priority = kPriorityMap.at(candidate->Name());
      if (priority < min_priority) {
        min_priority = priority;
        selected = candidate;
      }
    }
    MS_ASSERT(selected != nullptr);
    auto kernel = selected->CreateKernel(spec, inputs, outputs, ctx);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "Create kernel from " << selected->Name() << " failed, op_type: " << spec.op_type
                    << ", kernel attr: " << spec.attr;
    }
    return kernel;
  }
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_DEFAULT_KERNEL_SELECTOR_H
