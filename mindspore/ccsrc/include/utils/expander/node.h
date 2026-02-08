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

#ifndef MINDSPORE_CCSRC_UTILS_EXPANDER_NODE_H_
#define MINDSPORE_CCSRC_UTILS_EXPANDER_NODE_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "include/utils/visible.h"

namespace mindspore {
enum class InputType;
namespace expander {
class Emitter;
using DAttr = std::vector<std::pair<std::string, ValuePtr>>;

class COMMON_EXPORT Node : public std::enable_shared_from_this<Node> {
 public:
  explicit Node(Emitter *emitter);
  Node(Emitter *emitter, const ValuePtr &value) : emitter_(emitter), value_(value) {}
  virtual ~Node() = default;

  virtual const AnfNodePtr &get() const { MS_EXCEPTION(NotImplementedError) << "Base Node not implement get() method"; }

  virtual InputType input_type();
  virtual AbstractBasePtr abstract();

  void SetValue(const ValuePtr &val) { value_ = val; }
  ValuePtr Value() { return value_; }
  virtual ValuePtr BuildValue();
  virtual bool HasAbstractValue();
  virtual BaseShapePtr GetShape();
  virtual TypePtr GetType();

  virtual std::vector<int64_t> shape();
  std::vector<std::vector<int64_t>> shapes();
  virtual TypePtr dtype();
  std::vector<TypePtr> dtypes();
  Emitter *emitter() { return emitter_; }
  virtual std::string ToString() const;
  virtual void set_debug_info(const std::string &debug_info) {}
  virtual std::string debug_info() const { return ""; }
  virtual bool is_used_value() const {
    MS_EXCEPTION(NotImplementedError) << "Base Node not implement is_used_value() method";
  }
  virtual bool need_compute_grad_out() const { return true; }

 protected:
  // hold the emitter who created this node.
  Emitter *emitter_{nullptr};
  // cache the output shape after first query
  BaseShapePtr shape_{nullptr};
  // cache the output dtype after first query
  TypePtr type_{nullptr};
  // cache the value of node
  ValuePtr value_{nullptr};
};
using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;

class COMMON_EXPORT IrNode : public Node {
 public:
  IrNode(const AnfNodePtr anfnode, Emitter *emitter) : Node(emitter), anf_node_(anfnode) {}
  const AnfNodePtr &get() const override { return anf_node_; }
  InputType input_type() override;
  AbstractBasePtr abstract() override;

  ValuePtr BuildValue() override;
  bool HasAbstractValue() override;
  BaseShapePtr GetShape() override;
  TypePtr GetType() override;

  std::string ToString() const override;
  void set_debug_info(const std::string &debug_info) override;
  std::string debug_info() const override;
  bool is_used_value() const override { return is_used_value_; }

 private:
  // the wrapped anfnode.
  AnfNodePtr anf_node_{nullptr};
  // whether use value
  bool is_used_value_{false};
};
using IrNodePtr = std::shared_ptr<IrNode>;

class COMMON_EXPORT FuncNode : public Node {
 public:
  FuncNode(const ValuePtr &value, const abstract::AbstractBasePtr &abs, InputType input_type, Emitter *emitter)
      : Node(emitter, value), abstract_(abs), input_type_(input_type) {}
  ValuePtr BuildValue() override;
  InputType input_type() override;
  std::vector<int64_t> shape() override;
  TypePtr dtype() override;
  void set_node_type(InputType input_type) { input_type_ = input_type; }
  AbstractBasePtr abstract() override;
  void set_abstract(const AbstractBasePtr &abs) { abstract_ = abs; }
  BaseShapePtr GetShape() override;
  TypePtr GetType() override;
  std::string ToString() const override { return value_->ToString(); }
  void set_debug_info(const std::string &debug_info) override {}
  std::string debug_info() const override { return ""; }
  bool need_compute_grad_out() const override { return need_compute_grad_out_; }
  void set_need_compute_grad_out(bool need_compute_grad_out) { need_compute_grad_out_ = need_compute_grad_out; }

 private:
  AbstractBasePtr abstract_;
  InputType input_type_;
  bool need_compute_grad_out_{true};
};
using FuncNodePtr = std::shared_ptr<FuncNode>;
}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_EXPANDER_NODE_H_
