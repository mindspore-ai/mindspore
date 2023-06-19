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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_KERNEL_SELECTOR_FORMAT_FIRST_KERNEL_SELECTOR_H
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_KERNEL_SELECTOR_FORMAT_FIRST_KERNEL_SELECTOR_H

#include <vector>
#include <map>
#include <memory>
#include <string>
#include "src/extendrt/kernel/kernel_selector/kernel_selector.h"
#include "src/extendrt/kernel/nnacl/nnacl_lib.h"
#include "src/extendrt/kernel/default/default_kernel_lib.h"
#include "src/extendrt/graph_compiler/compile_option.h"

namespace mindspore::kernel {
class FormatFirstKernelSelector : public KernelSelector {
 public:
  explicit FormatFirstKernelSelector(const std::shared_ptr<lite::CompileOption> &compile_option)
      : KernelSelector(compile_option) {}
  ~FormatFirstKernelSelector() override = default;

  InferKernel *CreateKernel(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                            const std::vector<InferTensor *> &outputs, const InferContext *ctx) override {
    static std::map<std::string, int> kPriorityMap{
      {"custom", 0},
      {kNNACLLibName, 1},
      {"bolt", 1},
      {kDefaultKernelLibName, 100},
    };
    auto match_ks = spec;
    auto candidates = Candidates(match_ks.op_type, match_ks.attr, match_ks.backend, match_ks.format);
    if (candidates.empty()) {
      match_ks.format = DEFAULT_FORMAT;
      candidates = Candidates(match_ks.op_type, match_ks.attr, match_ks.backend, match_ks.format);
    }
    if (candidates.empty()) {
      MS_LOG(ERROR) << "Can not find suitable kernellib, op_type: " << spec.op_type << ", kernel attr: " << spec.attr;
      return nullptr;
    }
    int min_priority = INT32_MAX;
    const kernel::KernelLib *selected{nullptr};
    constexpr int default_priority = 1000;
    for (auto &candidate : candidates) {
      auto iter = kPriorityMap.find(candidate->Name());
      int priority = (iter == kPriorityMap.end()) ? default_priority : iter->second;
      if (priority < min_priority) {
        min_priority = priority;
        selected = candidate;
      }
    }
    MS_ASSERT(selected != nullptr);
    auto kernel = selected->CreateKernelExec(match_ks, inputs, outputs, ctx);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "Create kernel from " << selected->Name() << " failed, op_type: " << match_ks.op_type
                    << ", kernel attr: " << match_ks.attr;
      return nullptr;
    }
    MS_LOG(INFO) << "Create " << selected->Name() << " kernel for " << spec.cnode->fullname_with_scope();
    return kernel;
  }
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_KERNEL_SELECTOR_FORMAT_FIRST_KERNEL_SELECTOR_H
