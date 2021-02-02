/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_PATTERN_MATCHER_H_
#define MINDSPORE_CORE_IR_PATTERN_MATCHER_H_

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <algorithm>

#include "base/core_ops.h"
#include "ir/visitor.h"
#include "utils/shape_utils.h"

namespace mindspore {
///
///  Base class for all recognizable patterns.
///  We implement an Expression Template approach using static polymorphism based on
///  the Curiously Recurring Template Pattern (CRTP) which "achieves a similar effect
///  to the use of virtual functions without the costs..." as described in:
///  https://en.wikipedia.org/wiki/Expression_templates and
///  https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
///  The TryCapture function tries to capture the pattern with the given node.
///  The GetNode function builds a new node using the captured values.
///

template <typename T>
class PBase {
 public:
  bool CheckFunc(const PredicateFuncType &func, const AnfNodePtr &node) { return func(get_object().GetNode(node)); }

  const T &get_object() const { return *static_cast<const T *>(this); }

  template <typename TN>
  bool TryCapture(const TN &value) const {
    get_object().Reset();
    return get_object().TryCapture_(value);
  }

  using Internal = T;
};

template <typename T>
class PIsEqual {
 public:
  bool operator()(const T &lhs, const T &rhs) const { return lhs == rhs; }
};

template <typename T = AnfNodePtr>
class PatternNode : public PBase<PatternNode<T> > {
 public:
  T GetNode(const AnfNodePtr &) const {
    if (!captured_) {
      MS_EXCEPTION(ValueError) << "A Pattern wasn't captured for this Token before the call to GetNode.";
    }
    return captured_node_;
  }

  bool TryCapture_(const T &node) const {
    if (!captured_) {
      captured_node_ = node;
      captured_ = true;
      return true;
    }
    return PIsEqual<T>()(captured_node_, node);
  }

  void Reset() const { captured_ = false; }
  using Internal = const PatternNode<T> &;

 protected:
  mutable T captured_node_;
  mutable bool captured_{false};
};

template <typename T, typename T2>
class PBinOperation : public PBase<PBinOperation<T, T2> > {
 public:
  PBinOperation(const PrimitivePtr &prim, const T &x, const T2 &y, bool is_commutative = false)
      : prim_(prim), x_(x), y_(y), is_commutative_(is_commutative) {}
  ~PBinOperation() = default;

  AnfNodePtr GetNode(const AnfNodePtr &node) const {
    AnfNodePtr lhs = x_.GetNode(node);
    AnfNodePtr rhs = y_.GetNode(node);
    AnfNodePtrList list = {NewValueNode(prim_), lhs, rhs};
    return NewCNode(list, node->func_graph());
  }

  bool TryCapture_(const AnfNodePtr &node) const {
    if (IsPrimitiveCNode(node, prim_)) {
      auto cnode = node->cast<CNodePtr>();
      auto inputs = cnode->inputs();
      if (inputs.size() == 3) {
        // Binary Prim assumes only two inputs
        if (!x_.TryCapture(inputs[1]) || !y_.TryCapture(inputs[2])) {
          // If the operation is commutative, then check with inversed operands
          if (is_commutative_) {
            Reset();
            if (!x_.TryCapture(inputs[2]) || !y_.TryCapture(inputs[1])) {
              return false;
            }
            captured_binop_node_ = node;
            return true;
          }
          return false;
        }
        captured_binop_node_ = node;
        return true;
      }
    }
    return false;
  }

  /// Returns the original node captured by this Binary Operation Pattern.
  /// Throws exception if a node was not captured before.
  AnfNodePtr GetOriginalNode() const {
    if (captured_binop_node_ == nullptr) {
      MS_EXCEPTION(ValueError) << "A Node wasn't captured for this Pattern before attempting to get it.";
    }

    return captured_binop_node_;
  }

  void Reset() const {
    x_.Reset();
    y_.Reset();
    captured_binop_node_ = nullptr;
  }

  using Internal = const PBinOperation<T, T2> &;

 private:
  const PrimitivePtr prim_;
  typename T::Internal x_;
  typename T2::Internal y_;
  bool is_commutative_{false};
  mutable AnfNodePtr captured_binop_node_{nullptr};
};

template <typename T>
class PUnaryOperation : public PBase<PUnaryOperation<T> > {
 public:
  PUnaryOperation(const PrimitivePtr &prim, const T &x) : prim_(prim), x_(x) {}
  ~PUnaryOperation() = default;

  AnfNodePtr GetNode(const AnfNodePtr &node) const {
    AnfNodePtrList list = {NewValueNode(prim_), x_.GetNode(node)};
    return NewCNode(list, node->func_graph());
  }

  bool TryCapture_(const AnfNodePtr &node) const {
    if (IsPrimitiveCNode(node, prim_)) {
      auto cnode = node->cast<CNodePtr>();
      auto inputs = cnode->inputs();
      if (inputs.size() == 2 && x_.TryCapture(inputs[1])) {
        captured_unaryop_node_ = node;
        return true;
      }
    }
    return false;
  }

  AnfNodePtr GetOriginalNode() const {
    if (captured_unaryop_node_ == nullptr) {
      MS_EXCEPTION(ValueError) << "A Node wasn't captured for this Pattern before attempting to get it.";
    }
    return captured_unaryop_node_;
  }

  void Reset() const {
    x_.Reset();
    captured_unaryop_node_ = nullptr;
  }

  using Internal = const PUnaryOperation<T> &;

 private:
  const PrimitivePtr prim_;
  typename T::Internal x_;
  mutable AnfNodePtr captured_unaryop_node_{nullptr};
};

///
/// Helper functions to apply a pattern function on all elements of a tuple
///
namespace tuple_utils {
template <bool stop, size_t Index, typename Func>
struct apply_func_tuple_item {
  template <typename TTuple>
  static void apply(Func *func, const TTuple &tuple) {
    (*func)(Index, std::get<Index>(tuple));
    apply_func_tuple_item<(Index + 1) == std::tuple_size<TTuple>::value, (Index + 1), Func>::apply(func, tuple);
  }
};

template <size_t Index, typename Func>
struct apply_func_tuple_item<true, Index, Func> {
  template <typename TTuple>
  static void apply(Func *func, const TTuple &tuple) {}
};

template <typename Func, typename TTuple>
inline void apply_func_tuple(Func *func, const TTuple &tuple) {
  apply_func_tuple_item<std::tuple_size<TTuple>::value == 0, 0, Func>::apply(func, tuple);
}

struct PTupleResetCapture {
  template <typename T>
  void operator()(size_t i, const T &pattern) const {
    pattern.Reset();
  }
};

struct PTupleCapture {
  explicit PTupleCapture(const AnfNodePtrList tuple) : tuple_(tuple) {}

  template <typename TPattern>
  void operator()(size_t i, const TPattern &pattern) {
    // Check if the first node is a Primitive
    if (i == 0 && tuple_[i]->isa<Primitive>()) {
      auto prim = tuple_[i]->cast<PrimitivePtr>();
      if (tuple_[i] != pattern.GetNode(tuple_[i])) {
        captured_ = false;
      }
    } else {
      captured_ = captured_ && pattern.TryCapture_(tuple_[i]);
    }
  }

  const AnfNodePtrList tuple_;
  bool captured_{true};
};

struct PTupleGetNode {
  explicit PTupleGetNode(const AnfNodePtr &node) : node_(node) {}

  template <typename TPattern>
  void operator()(size_t, const TPattern &pattern) {
    args_.push_back(pattern.GetNode(node_));
  }

  const AnfNodePtr &node_;
  std::vector<AnfNodePtr> args_;
};
}  // namespace tuple_utils

template <typename... TArgs>
class PCNode : public PBase<PCNode<TArgs...> > {
 public:
  explicit PCNode(const TArgs &... args) : args_(args...) {}

  AnfNodePtr GetNode(const AnfNodePtr &node) const {
    tuple_utils::PTupleGetNode get_node(node);
    tuple_utils::apply_func_tuple(&get_node, args_);
    auto prim_cnode = get_node.args_;
    // In case this PCNode has captured extra nodes
    if (extra_nodes_.size() > 0) {
      prim_cnode.insert(prim_cnode.begin(), extra_nodes_.begin(), extra_nodes_.end());
    }
    return NewCNode(prim_cnode, node->func_graph());
  }

  bool TryCapture_(const AnfNodePtr &node) const {
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      auto inputs = cnode->inputs();

      auto pattern_arg_len = sizeof...(TArgs);
      // There aren't enough inputs in Node to fill up the Pattern
      if (inputs.size() < pattern_arg_len) {
        return false;
      }

      // Pattern must exactly match the number of Node inputs.
      if (!has_min_extra_nodes_) {
        // Inputs in Node perfectly match number of tokens in Pattern.
        if (inputs.size() == pattern_arg_len) {
          AnfNodePtrList tokens(inputs.begin(), inputs.end());
          tuple_utils::PTupleCapture capture_func(tokens);
          tuple_utils::apply_func_tuple(&capture_func, args_);
          return capture_func.captured_;
        }
        return false;
      }

      // Pattern may accept extra (non specified) nodes at the end of the CNode
      // There must be at least `min_extra_nodes` additional nodes in the inputs.
      if (inputs.size() >= pattern_arg_len + min_extra_nodes_) {
        AnfNodePtrList tokens(inputs.begin(), inputs.begin() + pattern_arg_len);
        tuple_utils::PTupleCapture capture_func(tokens);
        tuple_utils::apply_func_tuple(&capture_func, args_);
        // If it could capture the initial set of nodes specified in the Pattern
        // and there are enough extra inputs to add
        if (capture_func.captured_ && inputs.size() > pattern_arg_len) {
          extra_nodes_.insert(extra_nodes_.end(), inputs.begin() + pattern_arg_len, inputs.end());
          return true;
        }
        return capture_func.captured_;
      }
      return false;
    }
    return false;
  }

  /// This function sets the PCNode object to capture at least `min_extra_nodes_` nodes after the last one
  /// defined in the Pattern. e.g. `min_extra_nodes_ = 1` means the Pattern will be valid if there is one or
  /// more nodes after the last one specified when building the PCNode.
  const PCNode<TArgs...> &MinExtraNodes(const size_t &min_extra_nodes = 0) const {
    has_min_extra_nodes_ = true;
    min_extra_nodes_ = min_extra_nodes;
    return *this;
  }

  using Internal = const PCNode<TArgs...> &;

  void Reset() const {
    tuple_utils::PTupleResetCapture reset;
    tuple_utils::apply_func_tuple(&reset, args_);
    extra_nodes_.clear();
  }

 private:
  std::tuple<typename TArgs::Internal...> args_;
  mutable AnfNodePtrList extra_nodes_;
  mutable bool has_min_extra_nodes_{false};
  mutable size_t min_extra_nodes_{0};
};

template <typename... TArgs>
class PPrimitive : public PBase<PPrimitive<TArgs...> > {
 public:
  explicit PPrimitive(const PrimitivePtr &prim, const TArgs &... args) : prim_(prim), args_(args...) {}
  ~PPrimitive() = default;

  AnfNodePtr GetNode(const AnfNodePtr &node) const {
    tuple_utils::PTupleGetNode get_node(node);
    tuple_utils::apply_func_tuple(&get_node, args_);
    auto prim_cnode = get_node.args_;
    prim_cnode.insert(prim_cnode.begin(), NewValueNode(prim_));

    // In case this PPrimitive has captured extra nodes
    if (extra_nodes_.size() > 0) {
      prim_cnode.insert(prim_cnode.begin(), extra_nodes_.begin(), extra_nodes_.end());
    }
    return NewCNode(prim_cnode, node->func_graph());
  }

  bool TryCapture_(const AnfNodePtr &node) const {
    if (IsPrimitiveCNode(node, prim_)) {
      auto cnode = node->cast<CNodePtr>();
      auto inputs = cnode->inputs();
      // Number of arguments in Primitive Pattern (not including the Primitive node)
      auto pattern_arg_len = sizeof...(TArgs);
      // There aren't enough inputs in Node to fill up the Pattern
      if ((inputs.size() - 1) < pattern_arg_len) {
        return false;
      }

      // Pattern must exactly match the number of Node inputs.
      if (!has_min_extra_nodes_) {
        // Inputs in Node perfectly match number of tokens in Pattern.
        if ((inputs.size() - 1) == pattern_arg_len) {
          AnfNodePtrList tokens(inputs.begin() + 1, inputs.end());
          tuple_utils::PTupleCapture capture_func(tokens);
          tuple_utils::apply_func_tuple(&capture_func, args_);
          if (capture_func.captured_) {
            captured_prim_node_ = node;
          }
          return capture_func.captured_;
        }
        return false;
      }

      // Pattern may accept extra (non specified) nodes at the end of the Primitive
      // There must be at least `min_extra_nodes` additional nodes in the inputs.
      if ((inputs.size() - 1) >= pattern_arg_len + min_extra_nodes_) {
        AnfNodePtrList tokens(inputs.begin() + 1, inputs.begin() + 1 + pattern_arg_len);
        tuple_utils::PTupleCapture capture_func(tokens);
        tuple_utils::apply_func_tuple(&capture_func, args_);
        // If it could capture the initial set of nodes specified in the Pattern
        // and there are enough extra inputs to add
        if (capture_func.captured_) {
          captured_prim_node_ = node;
          if (inputs.size() > pattern_arg_len + 1) {
            extra_nodes_.insert(extra_nodes_.end(), inputs.begin() + 1 + pattern_arg_len, inputs.end());
          }
        }
        return capture_func.captured_;
      }
      return false;
    }
    return false;
  }

  /// This function sets the PPrimitive object to capture at least `min_extra_nodes_` nodes after the last one
  /// defined in the Pattern. e.g. `min_extra_nodes_ = 1` means the Pattern will be valid if there is one or
  /// more nodes after the last one specified when building the PPrimitive.
  const PPrimitive<TArgs...> &MinExtraNodes(const size_t &min_extra_nodes = 0) const {
    has_min_extra_nodes_ = true;
    min_extra_nodes_ = min_extra_nodes;
    return *this;
  }

  const AnfNodePtrList &GetCapturedExtraNodes() const { return extra_nodes_; }

  /// Returns the FuncGraph of the original node captured by this Primitive Pattern.
  /// Throws exception if a node was not captured before.
  FuncGraphPtr GetFuncGraph() const {
    if (captured_prim_node_ == nullptr) {
      MS_EXCEPTION(ValueError) << "A Node wasn't captured for this Pattern before attempting to get its FuncGraph.";
    }

    return captured_prim_node_->func_graph();
  }

  /// Returns the original node captured by this Primitive Pattern.
  /// Throws exception if a node was not captured before.
  AnfNodePtr GetOriginalNode() const {
    if (captured_prim_node_ == nullptr) {
      MS_EXCEPTION(ValueError) << "A Node wasn't captured for this Pattern before attempting to get it.";
    }

    return captured_prim_node_;
  }

  void Reset() const {
    tuple_utils::PTupleResetCapture reset;
    tuple_utils::apply_func_tuple(&reset, args_);
    extra_nodes_.clear();
    captured_prim_node_ = nullptr;
  }

  using Internal = const PPrimitive<TArgs...> &;

 private:
  const PrimitivePtr prim_;
  std::tuple<typename TArgs::Internal...> args_;
  mutable AnfNodePtrList extra_nodes_;
  mutable bool has_min_extra_nodes_{false};
  mutable size_t min_extra_nodes_{0};
  mutable AnfNodePtr captured_prim_node_{nullptr};
};

///
/// PConstant class can capture a value node of a specified value (check_value_)
/// or a non-specified one (any_value = true).
/// It can be configured to capture a scalar constant as well (is_scalar_ = true)
///
template <typename T = AnfNodePtr>
class PConstant : public PBase<PConstant<T> > {
 public:
  explicit PConstant(const AnfNodePtr &as_node, const bool any_value = true, const int64_t check_value = 0,
                     const bool is_scalar = false)
      : as_node_(as_node),
        captured_node_(as_node),
        any_value_(any_value),
        check_value_(check_value),
        is_scalar_(is_scalar) {}

  ~PConstant() = default;
  // Sets as_node_ as the node received as argument to produce a same-shape node with GetNode
  const PConstant<T> &WithShapeAs(const AnfNodePtr &node) const {
    if (node == nullptr) {
      MS_EXCEPTION(ValueError) << "WithShapeAs is trying to use a nullptr node.";
    }
    as_node_ = node;
    changed_shape_ = true;
    return *this;
  }

  // Sets as_node_ as the node caputred by the received Pattern token to produce a same-shape node with GetNode
  const PConstant<T> &WithShapeAs(const PatternNode<T> &pnode) const {
    if (captured_node_ == nullptr) {
      MS_EXCEPTION(ValueError) << "WithShapeAs is trying to use a Pattern token without previously capturing a node.";
    }
    as_node_ = pnode.GetNode(captured_node_);
    changed_shape_ = true;
    return *this;
  }

  /// Sets captured_node_ as the node captured by the Pattern received as argument
  /// to produce a new node with its contents when calling GetNode.
  const PConstant<T> &WithValueOf(const PatternNode<T> &pnode) const {
    if (!any_value_) {
      MS_EXCEPTION(ValueError) << "Must use a PConstant with `any_value = true` to use the value of another node.";
    }
    if (captured_node_ == nullptr) {
      MS_EXCEPTION(ValueError) << "WithValueOf is trying to use a Pattern token without previously capturing a node.";
    }
    captured_node_ = pnode.GetNode(captured_node_);
    changed_shape_ = true;
    return *this;
  }

  /// Create a new Value Node filled up with check_value.
  /// This function must be used immediately before GetNode to avoid replacing the expected result.
  /// Only valid for scalar constants. For tensors use WithShapeAs or WithValueOf.
  const PConstant<T> &NewValue() const {
    if (!is_scalar_) {
      MS_EXCEPTION(ValueError) << "NewValue is valid only for scalar PConstants.";
    }
    auto value_node_ = MakeValue(check_value_);
    captured_node_ = NewValueNode(value_node_);
    is_new_value_node_ = true;
    return *this;
  }

  AnfNodePtr GetNode(const AnfNodePtr &node) const {
    // If a NewValueNode was requested (using NewValue function) then return that created node.
    if (is_new_value_node_) {
      return captured_node_;
    }
    /// Return a NewTensorFilledWithData if the node was initialized to have a specific value
    /// even if it wasn't captured. Usually for zero constants (x - x => zero).
    /// If the shape was changed, use the new shape.
    if (changed_shape_ || !captured_) {
      if (!any_value_) {
        return NewTensorFilledWithData(as_node_, check_value_);
      }
      return NewTensorFilledWithData(as_node_, captured_node_);
    }
    return captured_node_;
  }

  bool TryCapture_(const AnfNodePtr &node) const {
    if (node->isa<ValueNode>()) {
      // If any_value_ is set don't check for the node's value. Just capture it.
      if (any_value_) {
        captured_node_ = node;
        captured_ = true;
        return true;
      }

      auto value = node->cast<ValueNodePtr>()->value();
      if ((is_scalar_ && IsTensorScalarConstant(value)) || (!is_scalar_ && IsTensorConstant(value))) {
        captured_node_ = node;
        captured_ = true;
        return true;
      }

      auto value_node_ = MakeValue(check_value_);
      if (*GetValueNode(node) == *value_node_) {
        captured_node_ = node;
        captured_ = true;
        return true;
      }
    }
    return false;
  }

  void Reset() const {
    captured_ = false;
    changed_shape_ = false;
    is_new_value_node_ = false;
  }

  // Support function used for checking if all values of a Tensor are equal to `check_value_`
  // Supported data types: double, float/float32, int/int32
  bool IsTensorConstant(const ValuePtr &value) const {
    if (!value->isa<tensor::Tensor>()) {
      return false;
    }
    auto tensor_ptr = dyn_cast<tensor::Tensor>(value);
    TypeId tensor_type = tensor_ptr->Dtype()->type_id();
    if ((tensor_type == TypeId::kNumberTypeFloat32) || (tensor_type == TypeId::kNumberTypeFloat)) {
      float *data2 = reinterpret_cast<float *>(tensor_ptr->data_c());
      auto threshold = FLT_MIN;
      for (int i = 0; i < tensor_ptr->DataSize(); i++) {
        if (fabs(data2[i] - check_value_) > threshold) {
          return false;
        }
      }
      return true;
    } else if (tensor_type == TypeId::kNumberTypeFloat64) {
      double *data2 = reinterpret_cast<double *>(tensor_ptr->data_c());
      auto threshold = DBL_MIN;
      for (int i = 0; i < tensor_ptr->DataSize(); i++) {
        if (fabs(data2[i] - check_value_) > threshold) {
          return false;
        }
      }
      return true;
    } else if ((tensor_type == TypeId::kNumberTypeInt32) || (tensor_type == TypeId::kNumberTypeInt)) {
      int *data2 = reinterpret_cast<int *>(tensor_ptr->data_c());
      for (int i = 0; i < tensor_ptr->DataSize(); i++) {
        if (data2[i] != check_value_) {
          return false;
        }
      }
      return true;
    }
    // Input Data Type is not supported
    return false;
  }

  bool IsTensorScalarConstant(const ValuePtr &value) const {
    if (!value->isa<tensor::Tensor>()) {
      return false;
    }
    auto tensor_ptr = dyn_cast<tensor::Tensor>(value);
    if ((tensor_ptr->DataSize() > 1) || (tensor_ptr->DataDim() > 0)) {
      return false;
    }
    return IsTensorConstant(value);
  }

  void *GetPointerToTensorData(const AnfNodePtr &node, bool writable = false) const {
    if (!node->isa<ValueNode>()) {
      return nullptr;
    }
    auto value = node->cast<ValueNodePtr>()->value();
    if (!value->isa<tensor::Tensor>()) {
      return nullptr;
    }

    tensor::TensorPtr tensor_ptr = dyn_cast<tensor::Tensor>(value);
    return tensor_ptr->data_c();
  }

  // Make a new tensor (when possible) with the same shape as of `node`
  // If x is nullptr then fill new tensor will "0"
  // If x is a tensor with empty shape then fill new tensor with the single value of x
  // If x is a tensor with same shape as `node` then return x as result
  AnfNodePtr NewTensorFilledWithData(const AnfNodePtr &node, const AnfNodePtr &x = nullptr) const {
    if ((node->abstract() == nullptr) || !node->abstract()->isa<abstract::AbstractTensor>()) {
      return nullptr;
    }

    auto tensor_abstract = node->abstract()->cast<abstract::AbstractTensorPtr>();
    TypePtr tensor_type_ptr = tensor_abstract->element()->BuildType();
    ShapeVector tensor_shape = tensor_abstract->shape()->shape();

    auto new_tensor_ptr = std::make_shared<tensor::Tensor>(tensor_type_ptr->type_id(), tensor_shape);
    size_t mem_size = GetTypeByte(tensor_type_ptr) * IntToSize(new_tensor_ptr->ElementsNum());
    char *data = reinterpret_cast<char *>(new_tensor_ptr->data_c());

    if (x == nullptr) {
      if (memset_s(data, mem_size, 0, mem_size) != 0) {
        return nullptr;
      }
      auto new_vnode = NewValueNode(new_tensor_ptr);
      new_vnode->set_abstract(new_tensor_ptr->ToAbstract());
      return new_vnode;
    }
    // x is not nullptr
    if (x->isa<CNode>() || x->isa<Parameter>()) {
      if ((x->abstract() == nullptr) || !x->abstract()->isa<abstract::AbstractTensor>()) {
        return nullptr;
      }
      auto x_abstract = x->abstract()->cast<abstract::AbstractTensorPtr>();
      ShapeVector x_shape = x_abstract->shape()->shape();
      if (x_shape != tensor_shape) {
        return nullptr;
      }
      return x;
    }

    if (!x->isa<ValueNode>()) {
      return nullptr;
    }
    auto x_value = x->cast<ValueNodePtr>()->value();
    if (!x_value->isa<tensor::Tensor>()) {
      return nullptr;
    }

    auto x_tensor_ptr = dyn_cast<tensor::Tensor>(x_value);
    if ((x_tensor_ptr->DataSize() > 1) && (x_tensor_ptr->DataSize() != new_tensor_ptr->DataSize())) {
      return nullptr;
    }
    int ret = 0;
    char *source_data = reinterpret_cast<char *>(GetPointerToTensorData(x));
    MS_EXCEPTION_IF_NULL(source_data);
    if (x_tensor_ptr->DataSize() == 1) {
      for (int i = 0; i < new_tensor_ptr->ElementsNum(); i++) {
        ret = memcpy_s(data + i * GetTypeByte(tensor_type_ptr), GetTypeByte(tensor_type_ptr), source_data,
                       GetTypeByte(tensor_type_ptr));
      }
    } else {
      ret = memcpy_s(data, mem_size, source_data, mem_size);
    }
    if (ret != 0) {
      MS_LOG(INFO) << "memcpy_s error, error no " << ret << ", source size " << mem_size << "dest size"
                   << new_tensor_ptr->DataSize();
      return nullptr;
    }
    auto new_vnode = NewValueNode(new_tensor_ptr);
    new_vnode->set_abstract(new_tensor_ptr->ToAbstract());
    return new_vnode;
  }

  AnfNodePtr NewTensorFilledWithData(const AnfNodePtr &node, const int &value) const {
    if ((node->abstract() == nullptr) || !node->abstract()->isa<abstract::AbstractTensor>()) {
      return nullptr;
    }

    auto tensor_abstract = node->abstract()->cast<abstract::AbstractTensorPtr>();
    TypePtr tensor_type_ptr = tensor_abstract->element()->BuildType();
    ShapeVector tensor_shape = tensor_abstract->shape()->shape();

    auto new_tensor_ptr = std::make_shared<tensor::Tensor>(tensor_type_ptr->type_id(), tensor_shape);
    size_t mem_size = GetTypeByte(tensor_type_ptr) * IntToSize(new_tensor_ptr->ElementsNum());
    char *data = reinterpret_cast<char *>(new_tensor_ptr->data_c());

    if (memset_s(data, mem_size, value, mem_size) != 0) {
      return nullptr;
    }
    auto new_vnode = NewValueNode(new_tensor_ptr);
    new_vnode->set_abstract(new_tensor_ptr->ToAbstract());
    return new_vnode;
  }

  template <typename TD>
  TD CalcuConstant(const TD &data, const PrimitivePtr &calcu_type) {
    TD tmp_data = data;
    if (calcu_type == prim::kPrimReciprocal) {
      if (data == 0) {
        MS_EXCEPTION(ValueError);
      } else {
        tmp_data = 1 / data;
      }
    }
    if (calcu_type == prim::kPrimNeg) {
      tmp_data = -data;
    }
    return tmp_data;
  }

  // calculate const with different operations
  AnfNodePtr ValueNodeWithOprations(const PrimitivePtr &calcu_type) {
    AnfNodePtr node = this->GetNode(captured_node_);
    if (!node->isa<ValueNode>()) {
      MS_EXCEPTION(ValueError) << "CalcuValue is trying to use a not ValueNode.";
    }
    auto value = node->cast<ValueNodePtr>()->value();
    if (value->isa<tensor::Tensor>()) {
      tensor::TensorPtr tensor_ptr = dyn_cast<tensor::Tensor>(value);
      TypeId tensor_type = tensor_ptr->Dtype()->type_id();
      auto tensor_abstract = node->abstract()->cast<abstract::AbstractTensorPtr>();
      TypePtr tensor_type_ptr = tensor_abstract->element()->BuildType();
      ShapeVector tensor_shape = tensor_abstract->shape()->shape();
      auto new_tensor_ptr = std::make_shared<tensor::Tensor>(tensor_type_ptr->type_id(), tensor_shape);
      size_t mem_size = GetTypeByte(tensor_type_ptr) * IntToSize(new_tensor_ptr->ElementsNum());
      if (new_tensor_ptr->DataSize() < tensor_ptr->DataSize()) {
        MS_EXCEPTION(ValueError) << "DataSize of new_tensor_ptr is smaller than DataSize of tensor_ptr";
      }
      if ((tensor_type == TypeId::kNumberTypeFloat32) || (tensor_type == TypeId::kNumberTypeFloat) ||
          (tensor_type == TypeId::kNumberTypeFloat64)) {
        float *data = reinterpret_cast<float *>(tensor_ptr->data_c());
        float *data2 = reinterpret_cast<float *>(new_tensor_ptr->data_c());
        if (memcpy_s(data2, mem_size, data, mem_size) != 0) {
          return nullptr;
        }
        for (int i = 0; i < tensor_ptr->DataSize(); i++) {
          if (data2[i] == 0 && calcu_type == prim::kPrimReciprocal) {
            return nullptr;
          }
          data2[i] = CalcuConstant(data2[i], calcu_type);
        }
      }
      if ((tensor_type == TypeId::kNumberTypeInt32) || (tensor_type == TypeId::kNumberTypeInt)) {
        int *data = reinterpret_cast<int *>(tensor_ptr->data_c());
        int *data2 = reinterpret_cast<int *>(new_tensor_ptr->data_c());
        if (memcpy_s(data2, mem_size, data, mem_size) != 0) {
          return nullptr;
        }
        for (int i = 0; i < tensor_ptr->DataSize(); i++) {
          if (data2[i] == 0 && calcu_type == prim::kPrimReciprocal) {
            return nullptr;
          }
          data2[i] = CalcuConstant(data2[i], calcu_type);
        }
      }
      if (tensor_type == TypeId::kNumberTypeFloat64) {
        double *data = reinterpret_cast<double *>(tensor_ptr->data_c());
        double *data2 = reinterpret_cast<double *>(new_tensor_ptr->data_c());
        if (memcpy_s(data2, mem_size, data, mem_size) != 0) {
          return nullptr;
        }
        for (int i = 0; i < tensor_ptr->DataSize(); i++) {
          if (data2[i] == 0 && calcu_type == prim::kPrimReciprocal) {
            return nullptr;
          }
          data2[i] = CalcuConstant(data2[i], calcu_type);
        }
      }
      auto new_vnode = NewValueNode(new_tensor_ptr);
      new_vnode->set_abstract(tensor_ptr->ToAbstract());
      return new_vnode;
    }
    return nullptr;
  }

  enum BinOperator {
    ADD = 0,
    MULTIPLY,
  };

  // Support function to add/multiply two constant tensors: partially support broadcasting shapes
  template <typename TM>
  void CalcByOperator(void *in_data_1, int in_data_1_size, void *in_data_2, int in_data_2_size, void **out_data,
                      int out_data_size, BinOperator bin_operator) const {
    if (out_data_size <= 0) {
      MS_EXCEPTION(ValueError) << "out_data_size should be greater than zeros";
    }
    TM *data_1 = reinterpret_cast<TM *>(in_data_1);
    TM *data_2 = reinterpret_cast<TM *>(in_data_2);
    TM *data_out = new TM[out_data_size];

    if (in_data_1_size == 1) {
      for (int i = 0; i < out_data_size; i++) {
        data_out[i] = data_1[0];
      }
    } else {
      for (int i = 0; i < out_data_size; i++) {
        data_out[i] = data_1[i];
      }
    }
    if (in_data_2_size == 1) {
      for (int i = 0; i < out_data_size; i++) {
        if (bin_operator == ADD) {
          data_out[i] += data_2[0];
        } else {
          data_out[i] *= data_2[0];
        }
      }
    } else {
      if (in_data_2_size < out_data_size) {
        MS_LOG(INFO) << "in_data_2_size:" << in_data_2_size << " is smaller than out_data_size:" << out_data_size
                     << ".in_data2 will be broadcast.";
      }
      auto min_size = std::min<int>(in_data_2_size, out_data_size);
      for (int i = 0; i < min_size; i++) {
        if (bin_operator == ADD) {
          data_out[i] += data_2[i];
        } else {
          data_out[i] *= data_2[i];
        }
      }
      // In case of in_data2_size < out_data_size
      for (int i = min_size; i < out_data_size; i++) {
        if (bin_operator != ADD) {
          // if operator is MUL, data_out[i] *= 0, => data_out[i] = 0.
          data_out[i] = 0;
        }
        // if operator is ADD, data_out[i] += 0, => data_out[i] = data_out[i], => NoChange.
      }
    }
    *out_data = reinterpret_cast<void *>(data_out);
    return;
  }

  AnfNodePtr AddByPatternConst(const PConstant<T> &vpnode_2, const AnfNodePtr &node_3) const {
    AnfNodePtr vnode_1 = this->GetNode(captured_node_);
    AnfNodePtr vnode_2 = vpnode_2.GetNode(captured_node_);
    return CalcConstantTensors(vnode_1, vnode_2, node_3, ADD);
  }

  AnfNodePtr MulByPatternConst(const PConstant<T> &vpnode_2, const AnfNodePtr &node_3) const {
    AnfNodePtr vnode_1 = this->GetNode(captured_node_);
    AnfNodePtr vnode_2 = vpnode_2.GetNode(captured_node_);
    return CalcConstantTensors(vnode_1, vnode_2, node_3, MULTIPLY);
  }

  AnfNodePtr CalcConstantTensors(const AnfNodePtr &vnode_1, const AnfNodePtr &vnode_2, const AnfNodePtr &node_3,
                                 BinOperator bin_operator) const {
    if (!vnode_1->isa<ValueNode>() || !vnode_2->isa<ValueNode>() || (vnode_1->abstract() == nullptr) ||
        (vnode_2->abstract() == nullptr) || (node_3->abstract() == nullptr)) {
      return nullptr;
    }

    auto value_1 = GetValueNode(vnode_1);
    auto value_2 = GetValueNode(vnode_2);
    if (!value_1->isa<tensor::Tensor>() || !value_2->isa<tensor::Tensor>()) {
      return nullptr;
    }

    auto tensor_ptr_1 = dyn_cast<tensor::Tensor>(value_1);
    auto tensor_ptr_2 = dyn_cast<tensor::Tensor>(value_2);

    auto tensor_1_abstract = vnode_1->abstract()->cast<abstract::AbstractTensorPtr>();
    auto tensor_2_abstract = vnode_1->abstract()->cast<abstract::AbstractTensorPtr>();
    TypePtr tensor_1_type_ptr = tensor_1_abstract->element()->BuildType();
    TypePtr tensor_2_type_ptr = tensor_2_abstract->element()->BuildType();

    ShapeVector tensor_out_shape;
    int data_out_size;
    tensor::TensorPtr new_tensor_ptr;

    if ((tensor_1_abstract->shape()->shape() == tensor_2_abstract->shape()->shape()) &&
        (tensor_1_type_ptr->type_id() == tensor_2_type_ptr->type_id())) {
      // If two constant nodes have the same shape, then create a new one with this shape
      tensor_out_shape = tensor_1_abstract->shape()->shape();
      data_out_size = std::accumulate(tensor_out_shape.begin(), tensor_out_shape.end(), 1, std::multiplies<int>());

      new_tensor_ptr = std::make_shared<tensor::Tensor>(tensor_1_type_ptr->type_id(), tensor_out_shape);
    } else {
      // If two constant nodes have different shapes, then create a new one node with the shape of the 3rd node
      auto tensor_3_abstract = node_3->abstract()->cast<abstract::AbstractTensorPtr>();

      TypePtr tensor_3_type_ptr = tensor_3_abstract->element()->BuildType();
      if ((tensor_1_type_ptr->type_id() != tensor_3_type_ptr->type_id()) ||
          (tensor_2_type_ptr->type_id() != tensor_3_type_ptr->type_id())) {
        return nullptr;
      }
      tensor_out_shape = tensor_3_abstract->shape()->shape();
      data_out_size = std::accumulate(tensor_out_shape.begin(), tensor_out_shape.end(), 1, std::multiplies<int>());
      if ((tensor_ptr_1->DataSize() > 1) && (tensor_ptr_1->DataSize() != data_out_size)) {
        return nullptr;
      }
      if ((tensor_ptr_2->DataSize() > 1) && (tensor_ptr_2->DataSize() != data_out_size)) {
        return nullptr;
      }
      new_tensor_ptr = std::make_shared<tensor::Tensor>(tensor_3_type_ptr->type_id(), tensor_out_shape);
    }

    size_t mem_size = GetTypeByte(new_tensor_ptr->Dtype()) * IntToSize(new_tensor_ptr->ElementsNum());
    char *data = reinterpret_cast<char *>(new_tensor_ptr->data_c());

    int ret = 0;
    void *data_out = nullptr;
    if ((new_tensor_ptr->data_type() == TypeId::kNumberTypeFloat32) ||
        (new_tensor_ptr->data_type() == TypeId::kNumberTypeFloat)) {
      CalcByOperator<float>(tensor_ptr_1->data_c(), tensor_ptr_1->DataSize(), tensor_ptr_2->data_c(),
                            tensor_ptr_2->DataSize(), &data_out, data_out_size, bin_operator);
      ret = memcpy_s(data, mem_size, data_out, mem_size);
      delete[] reinterpret_cast<float *>(data_out);
    } else {
      if (new_tensor_ptr->data_type() == TypeId::kNumberTypeFloat64) {
        CalcByOperator<double>(tensor_ptr_1->data_c(), tensor_ptr_1->DataSize(), tensor_ptr_2->data_c(),
                               tensor_ptr_2->DataSize(), &data_out, data_out_size, bin_operator);
        ret = memcpy_s(data, mem_size, data_out, mem_size);
        delete[] reinterpret_cast<double *>(data_out);
      } else {
        if ((new_tensor_ptr->data_type() == TypeId::kNumberTypeInt32) ||
            (new_tensor_ptr->data_type() == TypeId::kNumberTypeInt)) {
          CalcByOperator<int>(tensor_ptr_1->data_c(), tensor_ptr_1->DataSize(), tensor_ptr_2->data_c(),
                              tensor_ptr_2->DataSize(), &data_out, data_out_size, bin_operator);
          ret = memcpy_s(data, mem_size, data_out, mem_size);
          delete[] reinterpret_cast<int *>(data_out);
        } else {
          // Unsupported data types
          return nullptr;
        }
      }
    }
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret << ", source size " << mem_size << "dest size"
                        << new_tensor_ptr->DataSize();
    }
    auto new_vnode = NewValueNode(new_tensor_ptr);
    new_vnode->set_abstract(new_tensor_ptr->ToAbstract());
    return new_vnode;
  }

  using Internal = const PConstant<T> &;

 protected:
  mutable AnfNodePtr as_node_;
  mutable AnfNodePtr captured_node_;
  bool any_value_{true};
  int64_t check_value_{0};
  bool is_scalar_{false};
  mutable bool is_new_value_node_{false};
  mutable bool captured_{false};
  mutable bool changed_shape_{false};
};

// Macro for binary operation functions
#define BIN_OPERATION_PATTERN(Operator, MSPrimitive, Commutative)                   \
  template <typename T, typename T2>                                                \
  inline PBinOperation<T, T2> Operator(const PBase<T> &x, const PBase<T2> &y) {     \
    return PBinOperation(MSPrimitive, x.get_object(), y.get_object(), Commutative); \
  }

// Arithmetic operations
BIN_OPERATION_PATTERN(operator+, prim::kPrimAdd, true);
BIN_OPERATION_PATTERN(operator*, prim::kPrimMul, true);
BIN_OPERATION_PATTERN(operator/, prim::kPrimRealDiv, false);
BIN_OPERATION_PATTERN(operator-, prim::kPrimSub, false);

// Macros for match and replace
#define MATCH_REPLACE(OrigNode, CaptureNode, ReplaceWith) \
  if ((CaptureNode).TryCapture(OrigNode)) {               \
    auto rep = (ReplaceWith).GetNode(OrigNode);           \
    if (rep != nullptr) {                                 \
      return rep;                                         \
    }                                                     \
  }

#define MATCH_REPLACE_IF(OrigNode, CaptureNode, ReplaceWith, Condition) \
  if ((CaptureNode).TryCapture(OrigNode) && (Condition)) {              \
    auto rep = (ReplaceWith).GetNode(OrigNode);                         \
    if (rep != nullptr) {                                               \
      return rep;                                                       \
    }                                                                   \
  }

#define MATCH_REPLACE_IF_ELSE(OrigNode, CaptureNode, ReplaceWith, Condition, ElseNode) \
  if ((CaptureNode).TryCapture(OrigNode)) {                                            \
    if ((Condition)) {                                                                 \
      auto rep = (ReplaceWith).GetNode(OrigNode);                                      \
      if (rep != nullptr) {                                                            \
        return (ReplaceWith).GetNode(OrigNode);                                        \
      }                                                                                \
    } else {                                                                           \
      auto rep = (ElseNode).GetNode(OrigNode);                                         \
      if (rep != nullptr) {                                                            \
        return (ElseNode).GetNode(OrigNode);                                           \
      }                                                                                \
    }                                                                                  \
  }

#define MATCH_REPLACE_LAMBDA(OrigNode, CaptureNode, Lambda) \
  if ((CaptureNode).TryCapture(OrigNode)) {                 \
    auto rep = (Lambda)();                                  \
    if (rep != nullptr) {                                   \
      return rep;                                           \
    }                                                       \
  }

#define MATCH_REPLACE_LAMBDA_IF(OrigNode, CaptureNode, Lambda, Condition) \
  if ((CaptureNode).TryCapture(OrigNode) && (Condition)) {                \
    auto rep = (Lambda)();                                                \
    if (rep != nullptr) {                                                 \
      return rep;                                                         \
    }                                                                     \
  }

#define MATCH_REPLACE_LAMBDA_FLAG(OrigNode, CaptureNode, Lambda, Flag) \
  if ((CaptureNode).TryCapture(OrigNode)) {                            \
    auto rep = (Lambda)(Flag);                                         \
    if (rep != nullptr) {                                              \
      return rep;                                                      \
    }                                                                  \
  }
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_PATTERN_MATCHER_H_
