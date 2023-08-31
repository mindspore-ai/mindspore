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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TYPE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TYPE_H_

#include <memory>
#include <vector>
#include "include/api/delegate_api.h"
#include "ir/func_graph.h"
#include "src/extendrt/kernel/base_kernel.h"
#include "extendrt/kernel/kernel_lib.h"

namespace mindspore {
class ExtendDelegate : public IDelegate<FuncGraph, AnfNode, kernel::BaseKernel> {
 public:
  ExtendDelegate() = default;
  ~ExtendDelegate() override = default;

  void ReplaceNodes(const std::shared_ptr<FuncGraph> &graph) override {
    // not implemented
  }

  bool IsDelegateNode(const std::shared_ptr<AnfNode> &node) override {
    // not implemented
    return false;
  }

  std::shared_ptr<kernel::BaseKernel> CreateKernel(const std::shared_ptr<AnfNode> &node) override {
    // not implemented
    return nullptr;
  }

  virtual std::shared_ptr<kernel::BaseKernel> CreateKernel(const kernel::KernelSpec &spec,
                                                           const std::vector<InferTensor *> &inputs,
                                                           const std::vector<InferTensor *> &outputs,
                                                           const InferContext *ctx) const {
    // not implemented
    return nullptr;
  }
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TYPE_H_
