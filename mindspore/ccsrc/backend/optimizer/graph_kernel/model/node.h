/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <functional>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <set>
#include <iostream>
#include <utility>
#include <string>
#include <stdexcept>

#include "mindspore/core/ir/dtype/type_id.h"
#include "mindspore/core/ir/value.h"
#include "mindspore/core/ir/tensor.h"
#include "mindspore/core/utils/shape_utils.h"
#include "utils/utils.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace opt {
namespace graphkernel {
enum class NType {
  Base,
  Primitive,
  Parameter,
  Value,
  Output,
};

using DFormat = std::string;
using DShape = ShapeVector;
using DAttrs = std::unordered_map<std::string, ValuePtr>;

struct NodeBase {
  DShape shape;
  TypeId type;
  DFormat format;
};

class Node;
using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;
class Node : public NodeBase, public std::enable_shared_from_this<Node> {
 public:
  Node(const NodeBase &baseinfo, const std::string &name) : NodeBase(baseinfo), name_(name) {}
  virtual ~Node() {
    // remove this node from the previous nodes' user.
    SetInputs({});
  }

  void SetBaseInfo(NodeBase baseinfo) {
    this->shape = std::move(baseinfo.shape);
    this->type = std::move(baseinfo.type);
    this->format = std::move(baseinfo.format);
  }
  virtual NType NodeType() { return NType::Base; }
  friend std::ostream &operator<<(std::ostream &output, const Node &n) {
    std::ostringstream os;
    n.Dump(os);
    output << os.str();
    return output;
  }
  virtual void Dump(std::ostringstream &os) const = 0;
  virtual void DumpTensor(std::ostringstream &os) const;

  void AddInput(const NodePtr &new_input);
  void SetInput(size_t i, const NodePtr &new_input);
  void SetInputs(const NodePtrList &inputs);
  void ReplaceWith(const NodePtr &other_node);
  void SetAttrs(const DAttrs &attrs) { attrs_ = attrs; }
  void SetAttr(const std::string &key, const ValuePtr &value) { attrs_[key] = value; }

  template <typename T>
  std::shared_ptr<T> As() {
    return std::static_pointer_cast<T>(shared_from_this());
  }

  const std::string &name() const { return name_; }
  const DAttrs &attrs() const { return attrs_; }
  const NodePtr &input(size_t i) const { return inputs_[i]; }
  const NodePtrList &inputs() const { return inputs_; }
  const std::unordered_map<Node *, std::set<size_t>> &users() const { return users_; }

 protected:
  std::string name_;
  DAttrs attrs_;
  NodePtrList inputs_;
  std::unordered_map<Node *, std::set<size_t>> users_;

 private:
  // the nodes' users are only maintained by AddInput/SetInput.
  void AddUser(Node *user, size_t index) { users_[user].insert(index); }
  void RemoveUser(Node *user, size_t index) {
    if (auto iter = users_.find(user); iter != users_.end()) {
      iter->second.erase(index);
      if (iter->second.empty()) {
        users_.erase(iter);
      }
    }
  }
};

class ConstTensorNode : public Node {
 public:
  explicit ConstTensorNode(const tensor::TensorPtr &data, const std::string &name = "")
      : Node({data->shape(), data->data_type(), kOpFormat_DEFAULT}, name), data_(data) {}
  ~ConstTensorNode() = default;

  NType NodeType() override { return NType::Value; }
  void Dump(std::ostringstream &os) const override { os << ToString(); }
  void DumpTensor(std::ostringstream &os) const override { os << ToString(); }
  std::string ToString() const { return data_->data().ToString(this->type, this->shape, false); }
  const tensor::TensorPtr data() const { return data_; }

 protected:
  tensor::TensorPtr data_;
};

class ParamNode : public Node {
 public:
  ParamNode(const std::string &name, const NodeBase &baseinfo) : Node(baseinfo, name) {}
  ~ParamNode() = default;

  void Dump(std::ostringstream &os) const override { DumpTensor(os); }
  NType NodeType() override { return NType::Parameter; }
};

class OutputNode : public Node {
 public:
  OutputNode() : Node({{1}, TypeId::kNumberTypeBegin, kOpFormat_DEFAULT}, "Output") {}
  ~OutputNode() = default;

  void Dump(std::ostringstream &os) const override { ; }
  NType NodeType() override { return NType::Output; }
};
}  // namespace graphkernel
}  // namespace opt
}  // namespace mindspore
#endif
