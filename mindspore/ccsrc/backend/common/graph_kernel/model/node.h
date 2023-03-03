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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_NODE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_NODE_H_

#include <memory>
#include <vector>
#include <set>
#include <string>

#include "ir/dtype/type_id.h"
#include "ir/value.h"
#include "ir/tensor.h"
#include "utils/hash_map.h"
#include "utils/shape_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore::graphkernel::inner {
enum class NType {
  Base,
  Primitive,
  Parameter,
  Value,
  Output,
};

using DFormat = std::string;
using DShape = ShapeVector;
using DAttrs = mindspore::HashMap<std::string, ValuePtr>;

struct NodeBase {
  DShape shape;
  TypeId type;
  DFormat format;
};
using NodeBaseList = std::vector<NodeBase>;

class Node;
using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;
class Node : public NodeBase, public std::enable_shared_from_this<Node> {
 public:
  explicit Node(const NodeBase &baseinfo) : NodeBase(baseinfo) {}
  virtual ~Node() { ClearInputs(); }  // remove this node from the previous nodes' user.

  virtual NType NodeType() { return NType::Base; }
  virtual std::string ToString() const;
  virtual abstract::AbstractBasePtr ToAbstract() const;

  virtual void SetBaseInfo(const NodeBaseList &baseinfo);
  void AddInput(const NodePtr &new_input);
  void SetInput(size_t i, const NodePtr &new_input);
  void SetInputs(const NodePtrList &inputs);
  void ClearInputs() noexcept;
  void ReplaceWith(const NodePtr &other_node);
  void SetAttrs(const DAttrs &attrs) { attrs_ = attrs; }
  void SetAttr(const std::string &key, const ValuePtr &value) { attrs_[key] = value; }
  void SetDebugName(const std::string &debug_name) { debug_name_ = debug_name; }

  template <typename T>
  std::shared_ptr<T> As() {
    return std::static_pointer_cast<T>(shared_from_this());
  }

  const std::string &debug_name() const { return debug_name_; }
  const DAttrs &attrs() const { return attrs_; }
  const NodePtr &input(size_t i) const { return inputs_[i]; }
  const NodePtrList &inputs() const { return inputs_; }
  const mindspore::HashMap<Node *, std::set<size_t>> &users() const { return users_; }
  size_t tensor_size(bool in_bytes = false) const;
  const NodeBaseList &outputs() const { return outputs_; }

 protected:
  // only used in Dump function
  mutable std::string debug_name_;
  DAttrs attrs_;
  NodePtrList inputs_;
  // {user_node: {input edge index set}}
  mindspore::HashMap<Node *, std::set<size_t>> users_;
  // save output tensor info when the node is a multi-output operator.
  // it should keep empty when the node is single-output.
  NodeBaseList outputs_;

 private:
  // the nodes' users are only maintained by AddInput/SetInput.
  void AddUser(Node *const user, size_t index) { (void)users_[user].insert(index); }
  void RemoveUser(Node *const user, size_t index);
};

class ConstTensorNode : public Node {
 public:
  explicit ConstTensorNode(const tensor::TensorPtr &data)
      : Node({data->DataSize() == 1 ? DShape({1}) : data->shape(), data->data_type(), kOpFormat_DEFAULT}),
        data_(data) {}
  ~ConstTensorNode() = default;

  NType NodeType() override { return NType::Value; }
  std::string ToString() const override { return data_->data().ToString(data_->data_type(), data_->shape(), false); }
  const tensor::TensorPtr data() const { return data_; }
  abstract::AbstractBasePtr ToAbstract() const override { return data_->ToAbstract(); }

 protected:
  tensor::TensorPtr data_;
};

class ParamNode : public Node {
 public:
  explicit ParamNode(const NodeBase &baseinfo) : Node(baseinfo) {}
  ~ParamNode() = default;

  NType NodeType() override { return NType::Parameter; }
};

// the OutputNode's inputs are the real outputs of graph, like the `make_tuple` in FuncGraph.
class OutputNode : public Node {
 public:
  OutputNode() : Node({{1}, TypeId::kNumberTypeBegin, kOpFormat_DEFAULT}) {}
  ~OutputNode() = default;

  NType NodeType() override { return NType::Output; }
};
}  // namespace mindspore::graphkernel::inner
#endif
