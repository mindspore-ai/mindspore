/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_LOOP_UNROLL_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_LOOP_UNROLL_H_

#include <map>
#include <memory>

#include "ir/func_graph.h"
#include "primitive/sequence_ops.h"
#include "primitive/comparison_ops.h"
#include "primitive/framework_ops.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/pattern_matcher.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/jit/ps/parse/resolve.h"

namespace mindspore {
namespace opt {
namespace irpass {
// LoopUnroll for ops.Scan
class LoopUnrollBase : public AnfVisitor {
 public:
  explicit LoopUnrollBase(bool need_check_primJ, bool need_process)
      : need_check_primJ_(need_check_primJ), need_process_(need_process) {}
  ~LoopUnrollBase() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!NeedProcessScan(node)) {
      return nullptr;
    }
    CNodePtr scan_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(scan_cnode);
    const auto &inputs = scan_cnode->inputs();
    const size_t input_size = inputs.size();
    constexpr size_t expect_input_size = 6;
    MS_EXCEPTION_IF_CHECK_FAIL(input_size == expect_input_size, "Scan op has invalid input size.");
    auto loop_func_node = inputs[kIndex1];
    auto init_node = inputs[kIndex2];
    auto xs_node = inputs[kIndex3];
    auto length_node = inputs[kIndex4];
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto f_node = GetValueNode<FuncGraphPtr>(loop_func_node);
    MS_EXCEPTION_IF_NULL(f_node);

    bool is_none_node = IsValueNode<None>(xs_node);
    auto length_value_ptr = GetValueNode<Int64ImmPtr>(length_node);
    MS_EXCEPTION_IF_NULL(length_value_ptr);
    int64_t length_value = length_value_ptr->value();
    auto xs_abs = xs_node->abstract();
    MS_EXCEPTION_IF_NULL(xs_abs);
    PrimitivePtr getitem_op = prim::kPrimTupleGetItem;
    if (xs_abs->isa<abstract::AbstractList>()) {
      getitem_op = prim::kPrimListGetItem;
    }

    // Generate Loop FuncGraph for a single unrolled funcgraph call
    auto loop_unroll_fg = std::make_shared<FuncGraph>();
    AnfNodePtr loop_init = init_node;
    AnfNodePtr func_output = nullptr;
    AnfNodePtrList ys_result{NewValueNode(prim::kPrimMakeList)};
    for (int64_t i = 0; i < length_value; ++i) {
      AnfNodePtrList loop_inputs{{NewValueNode(f_node), loop_init}};
      if (is_none_node) {
        (void)loop_inputs.emplace_back(NewValueNode(static_cast<int64_t>(0)));
      } else {
        auto item =
          loop_unroll_fg->NewCNodeInOrder({NewValueNode(getitem_op), xs_node, NewValueNode(static_cast<int64_t>(i))});
        (void)loop_inputs.emplace_back(item);
      }
      func_output = loop_unroll_fg->NewCNodeInOrder(loop_inputs);
      loop_init = loop_unroll_fg->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), func_output, NewValueNode(static_cast<int64_t>(0))});
      auto new_y = loop_unroll_fg->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), func_output, NewValueNode(static_cast<int64_t>(1))});
      (void)ys_result.emplace_back(new_y);
    }
    auto loop_ys = loop_unroll_fg->NewCNodeInOrder(ys_result);
    auto output = loop_unroll_fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), loop_init, loop_ys});
    loop_unroll_fg->set_output(output);
    return fg->NewCNodeInOrder({NewValueNode(loop_unroll_fg)});
  }

  bool NeedProcessScan(const AnfNodePtr &node) {
    CNodePtr cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // For pass after automatic differentiation
    if (need_process_) {
      return true;
    }
    // For pass before automatic differentiation, check kPrimJ for the first time
    if (need_check_primJ_) {
      auto fg = node->func_graph();
      MS_EXCEPTION_IF_NULL(fg);
      auto manager = fg->manager();
      MS_EXCEPTION_IF_NULL(manager);
      const auto &nodes = manager->all_nodes();
      for (auto &cur_node : nodes) {
        if (IsPrimitiveCNode(cur_node, prim::kPrimJ)) {
          MS_LOG(DEBUG) << "Need Process loop unroll before automatic differentiation pass.";
          set_need_check_primJ();
          return true;
        }
      }
      MS_LOG(DEBUG) << "Need Process loop unroll after automatic differentiation pass.";
      set_need_check_primJ();
      return false;
    }
    return false;
  }

 private:
  bool need_check_primJ_;
  bool need_process_;
  void set_need_check_primJ() { need_check_primJ_ = false; }
};

class LoopUnrollBeforeGrad : public LoopUnrollBase {
 public:
  explicit LoopUnrollBeforeGrad(bool need_check_primJ = true, bool need_process = false)
      : LoopUnrollBase(need_check_primJ, need_process) {}
  ~LoopUnrollBeforeGrad() override = default;
};

class LoopUnrollAfterGrad : public LoopUnrollBase {
 public:
  explicit LoopUnrollAfterGrad(bool need_check_primJ = false, bool need_process = true)
      : LoopUnrollBase(need_check_primJ, need_process) {}
  ~LoopUnrollAfterGrad() override = default;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // #ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_LOOP_UNROLL_H_
