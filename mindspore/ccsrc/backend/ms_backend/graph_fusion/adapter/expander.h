/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/graph_fusion/core/expander.h"
#include "ir/func_graph.h"
#include <nlohmann/json.hpp>

namespace mindspore::graphkernel {
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

class UnfoldMakeTupleDeco : public ExpanderDecorator {
 public:
  explicit UnfoldMakeTupleDeco(const ExpanderPtr &decorated) : ExpanderDecorator(decorated) {}
  ~UnfoldMakeTupleDeco() override = default;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<UnfoldMakeTupleDeco>(decorated));
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

/**
 * Get the Expander which is used to expand a cnode to a funcgraph which composite same function with core ops.
 */
ExpanderPtr GetExpander(const AnfNodePtr &node, const ExpanderPtr &init);

/**
 * Get the Expander which is used to expand a cnode to a funcgraph which composite same function with core ops.
 */
ExpanderPtr GetExpander(const AnfNodePtr &node, bool abstract = true);

bool IsComplexOp(const AnfNodePtr &node);
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_ADAPTER_EXPANDER_H_
