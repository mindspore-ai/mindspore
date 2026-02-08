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
#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_CORE_EXPANDER_H_
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_CORE_EXPANDER_H_
#include <vector>
#include <memory>
#include "ir/func_graph.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_callback.h"

namespace mindspore::graphkernel {
class Expander {
 public:
  /**
   * Expand input cnode to a funcgraph which composite same function with core ops,
   * and return a cnode which input[0] is the funcgraph and input[1:-1] are inputs.
   */
  virtual AnfNodePtr Run(const AnfNodePtr &node) = 0;
  virtual ~Expander() = default;
};
using ExpanderPtr = std::shared_ptr<Expander>;

class DefaultExpander : public Expander {
 public:
  explicit DefaultExpander(const CallbackPtr &cb) : cb_(cb) {}
  ~DefaultExpander() override = default;
  AnfNodePtr Run(const AnfNodePtr &node) override;

 protected:
  virtual FuncGraphPtr ExpandToGraph(const CNodePtr &node);
  virtual AnfNodePtr CreateCallCNode(const FuncGraphPtr &fg, const CNodePtr &cnode);
  CallbackPtr cb_;
};

class LitegraphExpander : public DefaultExpander {
 public:
  explicit LitegraphExpander(const CallbackPtr &cb) : DefaultExpander(cb) {}
  ~LitegraphExpander() override = default;

 protected:
  FuncGraphPtr ExpandToGraph(const CNodePtr &node) override;
  AnfNodePtr CreateCallCNode(const FuncGraphPtr &sub_fg, const CNodePtr &cnode) override;
};

class ExpanderDecorator : public Expander {
 public:
  explicit ExpanderDecorator(const ExpanderPtr &decorated) : decorated_(decorated) {}
  ~ExpanderDecorator() override = default;
  /**
   * Do something before or after decoreated run.
   */
  AnfNodePtr Run(const AnfNodePtr &node) override;

 protected:
  // The expander cannot change the original node, this function clone the cnode with original info.
  CNodePtr QuickCloneCNode(const AnfNodePtr &node, bool clone_prim = false) const;

  ExpanderPtr decorated_;
};

using ExpanderCreatorFunc = std::function<ExpanderPtr(const ExpanderPtr &)>;
using ExpanderCreatorFuncList = std::vector<ExpanderCreatorFunc>;

// This decorator is required if we need to get the value of one input parameter during expanding
class DependValueDeco : public ExpanderDecorator {
 public:
  DependValueDeco(const ExpanderPtr &decorated, const HashSet<size_t> &input_idx)
      : ExpanderDecorator(decorated), input_idx_(input_idx) {}
  ~DependValueDeco() = default;

  static ExpanderCreatorFunc GetCreator(const HashSet<size_t> &input_idx) {
    return [input_idx](const ExpanderPtr &decorated) {
      return std::static_pointer_cast<Expander>(std::make_shared<DependValueDeco>(decorated, input_idx));
    };
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;

 protected:
  HashSet<size_t> input_idx_;
};
/**
 * Wrap Expander with decorators.
 */
ExpanderPtr WrapExpander(const ExpanderPtr &base, const ExpanderCreatorFuncList &deco_creators);
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_CORE_EXPANDER_H_
