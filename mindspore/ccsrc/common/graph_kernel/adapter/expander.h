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
#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_ADAPTER_EXPANDER_H_
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_ADAPTER_EXPANDER_H_
#include <vector>
#include <memory>
#include "common/graph_kernel/core/expander.h"
#include "ir/func_graph.h"
#include "include/common/visible.h"
#include <nlohmann/json.hpp>

namespace mindspore::graphkernel {
class PyExpander : public DefaultExpander {
 public:
  explicit PyExpander(const CallbackPtr &cb) : DefaultExpander(cb) {}
  virtual ~PyExpander() = default;

 protected:
  virtual bool CreateJsonInfo(const AnfNodePtr &node, nlohmann::json *kernel_json);
  FuncGraphPtr ExpandToGraph(const CNodePtr &node) override;
};

class ComplexOpDecorator : public ExpanderDecorator {
 public:
  explicit ComplexOpDecorator(const ExpanderPtr &decorated) : ExpanderDecorator(decorated) {}
  ~ComplexOpDecorator() override = default;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<ComplexOpDecorator>(decorated));
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;
};

/**
 * Get the Expander which is used to expand a cnode to a funcgraph which composite same function with core ops.
 */
COMMON_EXPORT ExpanderPtr GetExpander(const AnfNodePtr &, bool abstract = true);

/**
 * Inline the expanded func graph to main graph.
 */
COMMON_EXPORT void InlineExpandFuncGraph(const AnfNodePtr &expanding_node, const FuncGraphPtr &expanded_graph);

bool IsComplexOp(const AnfNodePtr &node);
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_ADAPTER_EXPANDER_H_
