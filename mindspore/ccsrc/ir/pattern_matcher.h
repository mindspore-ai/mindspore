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

#ifndef MINDSPORE_CCSRC_IR_PATTERN_MATCHER_H_
#define MINDSPORE_CCSRC_IR_PATTERN_MATCHER_H_

#include <tuple>
#include <vector>

#include "ir/anf.h"
#include "operator/ops.h"

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
  bool CheckFunc(const opt::PredicateFuncType &func, const AnfNodePtr &node) {
    return func(get_object().GetNode(node));
  }

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

template <typename T>
class PatternNode : public PBase<PatternNode<T> > {
 public:
  T GetNode(const AnfNodePtr &node) const {
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
  PBinOperation(const PrimitivePtr &prim, const T &x, const T2 &y) : prim_(prim), x_(x), y_(y) {}

  AnfNodePtr GetNode(const AnfNodePtr &node) const {
    AnfNodePtr lhs = x_.GetNode(node->func_graph());
    AnfNodePtr rhs = y_.GetNode(node->func_graph());
    AnfNodePtrList list = {prim_->cast<AnfNodePtr>(), lhs, rhs};
    return NewCNode(list, node->func_graph());
  }

  bool TryCapture_(const AnfNodePtr &node) const {
    if (IsPrimitiveCNode(node, prim_)) {
      auto cnode = node->cast<CNodePtr>();
      auto inputs = cnode->inputs();
      if (inputs.size() == 3) {
        // Binary Prim assumes only two inputs
        if (!x_.TryCapture_(inputs[1]) || !y_.TryCapture_(inputs[2])) {
          return false;
        }
        return true;
      }
    }
    return false;
  }

  void Reset() const {
    x_.Reset();
    y_.Reset();
  }

 private:
  const PrimitivePtr prim_;
  typename T::Internal x_;
  typename T2::Internal y_;
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
    return NewCNode(get_node.args_, node->func_graph());
  }

  bool TryCapture_(const AnfNodePtr &node) const {
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      auto inputs = cnode->inputs();
      if (inputs.size() != sizeof...(TArgs)) {
        return false;
      }
      tuple_utils::PTupleCapture capture_func(inputs);
      tuple_utils::apply_func_tuple(&capture_func, args_);
      return capture_func.captured_;
    }

    return false;
  }

  void Reset() const {
    tuple_utils::PTupleResetCapture reset;
    tuple_utils::apply_func_tuple(&reset, args_);
  }

 private:
  std::tuple<typename TArgs::Internal...> args_;
};

template <typename... TArgs>
class PPrimitive : public PBase<PPrimitive<TArgs...> > {
 public:
  explicit PPrimitive(const PrimitivePtr &prim, const TArgs &... args) : prim_(prim), args_(args...) {}

  AnfNodePtr GetNode(const AnfNodePtr &node) const {
    tuple_utils::PTupleGetNode get_node(node);
    tuple_utils::apply_func_tuple(&get_node, args_);
    auto prim_cnode = get_node.args_;
    prim_cnode.insert(prim_cnode.begin(), NewValueNode(prim_));
    return NewCNode(prim_cnode, node->func_graph());
  }

  bool TryCapture_(const AnfNodePtr &node) const {
    if (IsPrimitiveCNode(node, prim_)) {
      auto cnode = node->cast<CNodePtr>();
      auto inputs = cnode->inputs();
      if ((inputs.size() - 1) != sizeof...(TArgs)) {
        return false;
      }

      AnfNodePtrList rest(inputs.begin() + 1, inputs.end());
      tuple_utils::PTupleCapture capture_func(rest);
      tuple_utils::apply_func_tuple(&capture_func, args_);

      return capture_func.captured_;
    }

    return false;
  }

  void Reset() const {
    tuple_utils::PTupleResetCapture reset;
    tuple_utils::apply_func_tuple(&reset, args_);
  }

 private:
  const PrimitivePtr prim_;
  std::tuple<typename TArgs::Internal...> args_;
};

// Macro for binary operation functions
#define BIN_OPERATION_PATTERN(Operator, MSPrimitive)                            \
  template <typename T, typename T2>                                            \
  inline PBinOperation<T, T2> Operator(const PBase<T> &x, const PBase<T2> &y) { \
    return PBinOperation(MSPrimitive, x.get_object(), y.get_object());          \
  }

// Arithmetic operations
BIN_OPERATION_PATTERN(operator+, prim::kPrimTensorAdd);
BIN_OPERATION_PATTERN(operator*, prim::kPrimMul);

// Macros for match and replace
#define MATCH_REPLACE(OrigNode, CaptureNode, ReplaceWith) \
  if ((CaptureNode).TryCapture(OrigNode)) {               \
    return (ReplaceWith).GetNode(OrigNode);               \
  }

#define MATCH_REPLACE_IF(OrigNode, CaptureNode, ReplaceWith, Condition) \
  if ((CaptureNode).TryCapture(OrigNode) && (Condition)) {              \
    return (ReplaceWith).GetNode(OrigNode);                             \
  }

#define MATCH_REPLACE_IF_ELSE(OrigNode, CaptureNode, ReplaceWith, Condition, ElseNode) \
  if ((CaptureNode).TryCapture(OrigNode)) {                                            \
    if ((Condition)) {                                                                 \
      return (ReplaceWith).GetNode(OrigNode);                                          \
    }                                                                                  \
    return (ElseNode).GetNode(OrigNode);                                               \
  }

#define MATCH_REPLACE_LAMBDA(OrigNode, CaptureNode, Lambda) \
  if ((CaptureNode).TryCapture(OrigNode)) {                 \
    return (Lambda)();                                      \
  }

#define MATCH_REPLACE_LAMBDA_IF(OrigNode, CaptureNode, Lambda, Condition) \
  if ((CaptureNode).TryCapture(OrigNode) && (Condition)) {                \
    return (Lambda)();                                                    \
  }

}  // namespace mindspore

#endif  // #ifndef MINDSPORE_CCSRC_IR_PATTERN_MATCHER_H_
