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
  FuncGraphPtr ExpandToGraphByCallPyFn(const CNodePtr &node);
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

class ArgWithValueDeco : public ExpanderDecorator {
 public:
  explicit ArgWithValueDeco(const ExpanderPtr &decorated) : ExpanderDecorator(decorated) {}
  ~ArgWithValueDeco() override = default;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<ArgWithValueDeco>(decorated));
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;
};

class ProcessCustomOpDeco : public ExpanderDecorator {
 public:
  explicit ProcessCustomOpDeco(const ExpanderPtr &decorated) : ExpanderDecorator(decorated) {}
  ~ProcessCustomOpDeco() override = default;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<ProcessCustomOpDeco>(decorated));
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;
};

class SetDynamicShapeAttrDeco : public ExpanderDecorator {
 public:
  explicit SetDynamicShapeAttrDeco(const ExpanderPtr &decorated) : ExpanderDecorator(decorated) {}
  ~SetDynamicShapeAttrDeco() override = default;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<SetDynamicShapeAttrDeco>(decorated));
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;
};

class COMMON_EXPORT AttrToInputDeco : public ExpanderDecorator {
 public:
  AttrToInputDeco(const ExpanderPtr &decorated, bool is_backend)
      : ExpanderDecorator(decorated), is_backend_(is_backend) {}
  ~AttrToInputDeco() override = default;
  static ExpanderCreatorFunc GetCreator(bool is_backend) {
    return [is_backend](const ExpanderPtr &decorated) {
      return std::static_pointer_cast<Expander>(std::make_shared<AttrToInputDeco>(decorated, is_backend));
    };
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;

 protected:
  void ConvertAttrToInput(const FuncGraphPtr &graph);
  bool is_backend_{false};
};

/**
 * Get the Expander which is used to expand a cnode to a funcgraph which composite same function with core ops.
 */
COMMON_EXPORT ExpanderPtr GetExpander(const AnfNodePtr &node, bool abstract = true);

/**
 * Inline the expanded func graph to main graph.
 */
COMMON_EXPORT void InlineExpandFuncGraph(const AnfNodePtr &expanding_node, const FuncGraphPtr &expanded_graph);

/**
 * Try Expand cnode with check func.
 */
COMMON_EXPORT AnfNodePtr TryExpandCNode(const AnfNodePtr &node,
                                        const std::function<bool(const CNodePtr &kernel_node)> &func);

/**
 * Check if node can be expanded fallback.
 */
COMMON_EXPORT bool CanExpandFallback(const AnfNodePtr &node);

bool IsComplexOp(const AnfNodePtr &node);
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_ADAPTER_EXPANDER_H_
