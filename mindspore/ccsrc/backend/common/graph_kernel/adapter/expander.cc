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

#include "backend/common/graph_kernel/adapter/expander.h"

#include <map>
#include <set>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/random_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/python_adapter.h"
#include "kernel/graph_kernel/graph_kernel_json_generator.h"
#include "backend/common/graph_kernel/split_umonad.h"
#include "backend/common/graph_kernel/substitute_dropout.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/adapter/callback_impl.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/pass/inplace_assign_for_custom_op.h"
#include "kernel/common_utils.h"
#include "utils/ms_context.h"
#include "include/common/debug/anf_ir_dump.h"
#include "ir/func_graph_cloner.h"
#include "mindspore/core/ops/op_name.h"

namespace mindspore::graphkernel {
ExpanderPtr GetExpander(const AnfNodePtr &node, const ExpanderPtr &init) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(init);
  if (IsComplexOp(node)) {
    return ComplexOpDecorator::Creator(init);
  }

  constexpr size_t kAssignInputIdx = 1;
  constexpr size_t kLambOptimizerInputIdx = 12;
  constexpr size_t kLambWeightInputIdx = 4;
  constexpr size_t kRandomInputIdx = 1;
  constexpr size_t kAdamInputIdx = 10;
  constexpr size_t kAdamWeightDecayInputIdx = 9;
  std::map<std::string, ExpanderCreatorFuncList> creators = {
    {prim::kPrimAssignAdd->name(), {OpUMonadExpanderDeco::GetCreator(kAssignInputIdx)}},
    {prim::kLambApplyOptimizerAssign->name(), {OpUMonadExpanderDeco::GetCreator(kLambOptimizerInputIdx)}},
    {prim::kLambApplyWeightAssign->name(), {OpUMonadExpanderDeco::GetCreator(kLambWeightInputIdx)}},
    {prim::kPrimStandardNormal->name(), {OpUMonadExpanderDeco::GetCreator(kRandomInputIdx)}},
    {prim::kPrimAdam->name(), {OpUMonadExpanderDeco::GetCreator(kAdamInputIdx)}},
    {prim::kPrimAdamWeightDecay->name(), {OpUMonadExpanderDeco::GetCreator(kAdamWeightDecayInputIdx)}},
    {prim::kPrimDropout->name(), {DropoutExpanderDeco::Creator}},
    {prim::kPrimArgMaxWithValue->name(), {ArgWithValueDeco::Creator}},
    {prim::kPrimArgMinWithValue->name(), {ArgWithValueDeco::Creator}},
    {prim::kPrimSolveTriangular->name(), {ProcessCustomOpDeco::Creator}},
    {prim::kPrimLU->name(), {ProcessCustomOpDeco::Creator}},
    {prim::kPrimExpandDims->name(), {DependValueDeco::GetCreator({1})}},
    {prim::kPrimReduceMean->name(), {DependValueDeco::GetCreator({1})}},
    {prim::kPrimTile->name(), {DependValueDeco::GetCreator({1})}},
    {prim::kPrimSlice->name(), {DependValueDeco::GetCreator({1, 2})}},
    {prim::kPrimGather->name(), {DependValueDeco::GetCreator({2})}},
    {prim::kPrimAddN->name(), {UnfoldMakeTupleDeco::Creator}}};

  ExpanderPtr expander = init;
  const auto iter = creators.find(GetCNodePrimitive(node)->name());
  if (iter != creators.end()) {
    expander = WrapExpander(expander, iter->second);
  }
  if (common::AnfAlgo::IsDynamicShape(node)) {
    MS_LOG(INFO) << "try expander dynamic shape node.";
    expander = SetDynamicShapeAttrDeco::Creator(expander);
  }
  return expander;
}

ExpanderPtr GetExpander(const AnfNodePtr &node, bool abstract) {
  ExpanderPtr expander =
    abstract
      ? std::make_shared<LitegraphExpander>(
          std::static_pointer_cast<Callback>(std::make_shared<CallbackImplWithInferShape>()))
      : std::make_shared<LitegraphExpander>(std::static_pointer_cast<Callback>(std::make_shared<CallbackImpl>()));
  return GetExpander(node, expander);
}

bool CanExpandFallback(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  if (common::GetEnv("MS_DEV_EXPANDER_FALLBACK") == "off") {
    return false;
  }
  // Operators with 'batch_rank' attribute, which only appears in the vmap scenario, are not supported currently.
  if (common::AnfAlgo::HasNodeAttr(ops::kBatchRank, node->cast<CNodePtr>())) {
    return false;
  }
  static const std::vector<OpWithLevel> expander_fallback_ops_with_level = {
    {kAllTarget, OpLevel_0, prim::kPrimEqualCount},
    {kAllTarget, OpLevel_0, prim::kPrimSoftsign},
    {kAllTarget, OpLevel_0, prim::kPrimSquare},
    {kAllTarget, OpLevel_0, prim::kPrimBiasAdd},
    {kAllTarget, OpLevel_0, prim::kPrimReLU},
    {kAllTarget, OpLevel_0, prim::kPrimRelu},
    {kAllTarget, OpLevel_0, prim::kPrimSigmoid},
    {kAllTarget, OpLevel_0, prim::kPrimBiasAdd},
    {kAllTarget, OpLevel_0, prim::kPrimReLU},
    {kAllTarget, OpLevel_0, prim::kPrimSoftplus},
    {kAllTarget, OpLevel_0, prim::kPrimSoftplusGrad},
    {kAllTarget, OpLevel_0, prim::kPrimAssignAdd},
    {kAllTarget, OpLevel_0, prim::kLambApplyOptimizerAssign},
    {kAllTarget, OpLevel_0, prim::kLambApplyWeightAssign},
    {kAllTarget, OpLevel_0, prim::kPrimAdamWeightDecay},
    {kAllTarget, OpLevel_0, prim::kPrimStandardNormal},
    {kAllTarget, OpLevel_0, prim::kPrimAdam},
    // some ops including custom op are only used expand fallbak on Ascend.
    {kAscendDevice, OpLevel_0, prim::kPrimSolveTriangular},
    {kAscendDevice, OpLevel_0, prim::kPrimLU},
    // disabled
    {kAllTarget, OpLevel_1, prim::kPrimAddN},
    {kAllTarget, OpLevel_1, prim::kPrimErfc},
    {kAllTarget, OpLevel_1, prim::kPrimExpandDims},
    {kAllTarget, OpLevel_1, prim::kPrimGeLU},
    {kAllTarget, OpLevel_1, prim::kPrimGeLUGrad},
    {kAllTarget, OpLevel_1, prim::kPrimSqrtGrad},
    {kAllTarget, OpLevel_1, prim::kPrimTile},
    {kAllTarget, OpLevel_1, prim::kPrimClipByNormNoDivSum},
    {kAllTarget, OpLevel_1, prim::kSoftmaxGradExt},
    {kAllTarget, OpLevel_1, prim::kFusedMulAdd},
    {kAllTarget, OpLevel_1, prim::kPrimBatchMatMul},
    {kAllTarget, OpLevel_1, prim::kPrimBiasAddGrad},
    {kAllTarget, OpLevel_1, prim::kPrimDropout},
    {kAllTarget, OpLevel_1, prim::kPrimDropoutGrad},
    {kAllTarget, OpLevel_1, prim::kPrimMaximumGrad},
    {kAllTarget, OpLevel_1, prim::kPrimMinimumGrad},
    {kAllTarget, OpLevel_1, prim::kPrimLayerNorm},
    {kAllTarget, OpLevel_1, prim::kPrimLayerNormGrad},
    {kAllTarget, OpLevel_1, prim::kPrimLogSoftmax},
    {kAllTarget, OpLevel_1, prim::kPrimLogSoftmaxV2},
    {kAllTarget, OpLevel_1, prim::kPrimLogSoftmaxGrad},
    {kAllTarget, OpLevel_1, prim::kPrimMatMul},
    {kAllTarget, OpLevel_1, prim::kPrimReduceMean},
    {kAllTarget, OpLevel_1, prim::kPrimReluGrad},
    {kAllTarget, OpLevel_1, prim::kPrimSigmoidGrad},
    {kAllTarget, OpLevel_1, prim::kPrimSigmoidCrossEntropyWithLogits},
    {kAllTarget, OpLevel_1, prim::kPrimSigmoidCrossEntropyWithLogitsGrad},
    {kAllTarget, OpLevel_1, prim::kPrimSlice},
    {kAllTarget, OpLevel_1, prim::kPrimSoftmax},
    {kAllTarget, OpLevel_1, prim::kPrimSoftmaxV2},
    {kAllTarget, OpLevel_1, prim::kPrimSoftmaxCrossEntropyWithLogits},
    {kAllTarget, OpLevel_1, prim::kPrimSquaredDifference},
    {kAllTarget, OpLevel_1, prim::kPrimSqueeze},
    {kAllTarget, OpLevel_1, prim::kPrimSquareSumAll},
    {kAllTarget, OpLevel_1, prim::kPrimIdentityMath},
    {kAllTarget, OpLevel_1, prim::kPrimOnesLike},
    {kAllTarget, OpLevel_1, prim::kPrimBiasAddGrad},
    {kAllTarget, OpLevel_1, prim::kPrimMaximumGrad},
    {kAllTarget, OpLevel_1, prim::kPrimMinimumGrad},
    {kAllTarget, OpLevel_1, prim::kPrimTanhGrad},
  };
  unsigned int op_level = (common::GetEnv("MS_DEV_EXPANDER_FALLBACK") == "1") ? 1 : 0;
  auto ops = GkUtils::GetValidOps(expander_fallback_ops_with_level, op_level, {}, {}, {});
  return std::any_of(ops.begin(), ops.end(),
                     [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
}

AnfNodePtr ProcessCustomOpDeco::Run(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto new_node = decorated_->Run(node);
  auto graph = GetCNodeFuncGraph(new_node);
  if (graph == nullptr) {
    return nullptr;
  }
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InplaceAssignForCustomOp>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  return new_node;
}

AnfNodePtr TryExpandCNode(const AnfNodePtr &node, const std::function<bool(const CNodePtr &)> &func) {
  if (!CanExpandFallback(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  auto expander = GetExpander(node);
  auto res = expander->Run(node);
  auto expand_fg = GetCNodeFuncGraph(res);
  if (expand_fg == nullptr) {
    return nullptr;
  }
  // For Ascend, the selectkernel function may check and change the input Parameter of kernel,
  // so we replace the inner Parameter to outer Parameter
  bool need_replace_parameter = Callback::Instance()->GetTargetFromContext() == kAscendDevice;
  std::map<AnfNodePtr, AnfNodePtr> param_map;
  if (need_replace_parameter) {
    auto &params = expand_fg->parameters();
    for (size_t i = 0; i < params.size(); i++) {
      if (cnode->input(i + 1)->isa<Parameter>()) {
        param_map[params[i]] = cnode->input(i + 1);
      }
    }
    need_replace_parameter = !param_map.empty();
  }

  auto todos = TopoSort(expand_fg->output());
  for (const auto &inner_node : todos) {
    if (!AnfUtils::IsRealCNodeKernel(inner_node)) {
      continue;
    }
    bool suc = false;
    auto inner_cnode = inner_node->cast<CNodePtr>();
    if (need_replace_parameter) {
      std::vector<std::pair<size_t, AnfNodePtr>> ori_input;
      for (size_t i = 1; i < inner_cnode->size(); i++) {
        auto iter = param_map.find(inner_cnode->input(i));
        if (iter != param_map.end()) {
          MS_LOG(DEBUG) << "Replace " << inner_cnode->input(i)->DebugString() << " by " << iter->second->DebugString();
          (void)ori_input.emplace_back(i, inner_cnode->input(i));
          inner_cnode->set_input(i, iter->second);
        }
      }
      suc = func(inner_cnode);
      // recover the origin inputs
      for (auto &ori : ori_input) {
        inner_cnode->set_input(ori.first, ori.second);
      }
    } else {
      suc = func(inner_cnode);
    }
    if (!suc) {
      MS_LOG(DEBUG) << "Expanding core ops [" << inner_node->fullname_with_scope() << "] failed.";
      res = nullptr;
      break;
    }
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kAdvanced)) {
    DumpIR("verbose_ir_files/expand_" + GetCNodeFuncName(node->cast<CNodePtr>()) + ".ir", expand_fg);
  }
#endif
  return res;
}

void SetDynamicShapeAttrToCNode(const CNodePtr &cnode) {
  auto in_dynamic = common::AnfAlgo::IsNodeInputDynamicShape(cnode);
  auto out_dynamic = common::AnfAlgo::IsNodeOutputDynamicShape(cnode);
  if (in_dynamic && !common::AnfAlgo::HasNodeAttr(kAttrInputIsDynamicShape, cnode)) {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), cnode);
  }
  if (out_dynamic && !common::AnfAlgo::HasNodeAttr(kAttrOutputIsDynamicShape, cnode)) {
    common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), cnode);
  }
}

void SetDynamicShapeAttr(const FuncGraphPtr &graph) {
  auto todos = TopoSort(graph->get_return());
  for (const auto &node : todos) {
    if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = dyn_cast<CNode>(node);
    SetDynamicShapeAttrToCNode(cnode);
  }
}

AnfNodePtr SetDynamicShapeAttrDeco::Run(const AnfNodePtr &node) {
  auto new_node = decorated_->Run(node);
  if (new_node == nullptr) {
    return nullptr;
  }
  auto new_cnode = dyn_cast<CNode>(new_node);
  auto expand_fg = GetCNodeFuncGraph(new_cnode);
  SetDynamicShapeAttr(expand_fg);
  new_cnode->set_input(0, NewValueNode(expand_fg));
  return new_cnode;
}

AnfNodePtr ComplexOpDecorator::Run(const AnfNodePtr &node) {
  auto cnode = QuickCloneCNode(node);
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  cnode->set_input(0, NewValueNode(std::make_shared<Primitive>("C" + prim->name(), prim->attrs())));
  return decorated_->Run(cnode);
}

// Used for ArgMaxWithValue(ArgMinWithValue) which output is tuple(index,value)
// Currently only expand it when output[1] has users and output[0] has no users
// In this case, ArgMaxWithValue(ArgMinWithValue) can be converted to ReduceMax(ReduceMin)
// If output[0] has users, expanding is not allowed
AnfNodePtr ArgWithValueDeco::Run(const AnfNodePtr &node) {
  auto mng = GkUtils::GetFuncGraphManager(node->func_graph());
  bool res = false;
  if (auto iter = mng->node_users().find(node); iter != mng->node_users().end()) {
    auto output_info_list = iter->second;
    res = std::all_of(output_info_list.begin(), output_info_list.end(), [](const std::pair<AnfNodePtr, int> &info) {
      if (IsPrimitiveCNode(info.first, prim::kPrimTupleGetItem)) {
        const auto &cnode = info.first->cast<CNodePtr>();
        auto value_ptr = GetValueNode(cnode->input(kInputNodeOutputIndexInTupleGetItem));
        MS_EXCEPTION_IF_NULL(value_ptr);
        return GetValue<int64_t>(value_ptr) == 1;
      }
      return false;
    });
  }
  return res ? decorated_->Run(node) : nullptr;
}

AnfNodePtr UnfoldMakeTupleDeco::Run(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() == kIndex2 && IsPrimitiveCNode(cnode->input(1), prim::kPrimMakeTuple)) {
    auto make_tupe_cnode = cnode->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tupe_cnode);
    std::vector<AnfNodePtr> new_inputs;
    new_inputs.push_back(cnode->input(0));
    for (size_t i = 1; i < make_tupe_cnode->inputs().size(); ++i) {
      new_inputs.push_back(make_tupe_cnode->input(i));
    }
    cnode = QuickCloneCNode(cnode);
    cnode->set_inputs(new_inputs);
  }
  return decorated_->Run(cnode);
}

void InlineExpandFuncGraph(const AnfNodePtr &expanding_node, const FuncGraphPtr &expanded_graph) {
  auto main_graph = expanding_node->func_graph();
  auto mng = main_graph->manager();
  if (mng == nullptr) {
    mng = Manage(main_graph, true);
    main_graph->set_manager(mng);
  }
  auto cnode = expanding_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfNodePtrList inp(cnode->inputs().begin() + 1, cnode->inputs().end());
  auto out = InlineClone(expanded_graph, main_graph, inp, cnode->input(0)->scope(), cnode->debug_info());
  (void)mng->Replace(expanding_node, out);
}

bool IsComplexOp(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t i = 1; i < cnode->size(); i++) {
    auto input = cnode->input(i);
    TypePtr input_type = input->Type();
    if (input_type == nullptr || !input_type->isa<TensorType>()) {
      return false;
    }
    input_type = input_type->cast<TensorTypePtr>()->element();
    if (input_type->type_id() == kNumberTypeComplex64 || input_type->type_id() == kNumberTypeComplex128) {
      return true;
    }
  }
  return false;
}
}  // namespace mindspore::graphkernel
