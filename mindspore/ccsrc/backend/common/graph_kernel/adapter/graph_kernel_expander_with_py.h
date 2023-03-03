/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_EXPANDER_WITH_PY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_EXPANDER_WITH_PY_H_
#include <memory>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "backend/common/graph_kernel/core/graph_kernel_expander.h"
#include "backend/common/graph_kernel/adapter/expander.h"
#include "ir/func_graph.h"

namespace mindspore::graphkernel {
class GraphKernelExpanderWithPy : public GraphKernelExpander {
 public:
  GraphKernelExpanderWithPy() : GraphKernelExpander() {}
  explicit GraphKernelExpanderWithPy(const std::string &name) : GraphKernelExpander(name) {}
  ~GraphKernelExpanderWithPy() override = default;
  static std::vector<PrimitivePtr> GetExpanderOps();

 protected:
  std::vector<PrimitivePtr> InitOpList() override;
  ExpanderPtr InitExpander(const AnfNodePtr &node) override;
  bool CanExpand(const CNodePtr &node) const override {
    if (IsComplexOp(node)) {
      return true;
    }
    return GraphKernelExpander::CanExpand(node);
  }
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_EXPANDER_WITH_PY_H_
