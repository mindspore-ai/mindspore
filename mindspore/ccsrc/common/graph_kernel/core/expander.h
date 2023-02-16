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
#include "include/backend/visible.h"
#include "common/graph_kernel/core/graph_kernel_callback.h"

namespace mindspore::graphkernel {
class BACKEND_EXPORT Expander {
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
  virtual ~DefaultExpander() = default;
  AnfNodePtr Run(const AnfNodePtr &node) override;

 protected:
  virtual FuncGraphPtr ExpandToGraph(const CNodePtr &node);
  CallbackPtr cb_;
};

class BACKEND_EXPORT ExpanderDecorator : public Expander {
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

class BACKEND_EXPORT InputToAttrDeco : public ExpanderDecorator {
 public:
  explicit InputToAttrDeco(const ExpanderPtr &decorated) : ExpanderDecorator(decorated) {}
  ~InputToAttrDeco() = default;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<InputToAttrDeco>(decorated));
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;
};

/**
 * Wrap Expander with decorators.
 */
BACKEND_EXPORT ExpanderPtr WrapExpander(const ExpanderPtr &base, const ExpanderCreatorFuncList &deco_creators);
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_CORE_EXPANDER_H_
