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

#ifndef MINDSPORE_CCSRC_COMMON_EXPANDER_CORE_NODE_H_
#define MINDSPORE_CCSRC_COMMON_EXPANDER_CORE_NODE_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "include/common/visible.h"

namespace mindspore {
namespace expander {
class Emitter;
using DAttr = std::vector<std::pair<std::string, ValuePtr>>;

class COMMON_EXPORT Node : public std::enable_shared_from_this<Node> {
 public:
  Node(const AnfNodePtr &node, Emitter *emitter);
  ~Node() = default;

  const AnfNodePtr &get() const { return anf_node_; }

  template <typename T>
  bool isa() const {
    return anf_node_->isa<T>();
  }
  template <typename T>
  T get() const {
    return anf_node_->cast<T>();
  }

  AbstractBasePtr abstract();

  void SetValue(const ValuePtr &val) { value_ = val; }
  ValuePtr Value() { return value_; }
  ValuePtr BuildValue();
  bool HasAbstractValue();

  std::vector<int64_t> shape();
  std::vector<std::vector<int64_t>> shapes();

  TypePtr dtype();
  std::vector<TypePtr> dtypes();

  Emitter *emitter() const { return emitter_; }
  bool is_used_value() const { return is_used_value_; }

 protected:
  // the wrapped anfnode.
  AnfNodePtr anf_node_{nullptr};
  // hold the emitter who created this node.
  Emitter *emitter_{nullptr};

  // cache the output shape after first query
  BaseShapePtr shape_{nullptr};
  // cache the output dtype after first query
  TypePtr type_{nullptr};
  // cache the value of node
  ValuePtr value_{nullptr};
  // whether use value
  bool is_used_value_{false};
};
using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;
}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_EXPANDER_CORE_NODE_H_
