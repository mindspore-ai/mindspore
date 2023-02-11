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

#include "common/graph_kernel/adapter/expander.h"

#include <map>
#include <set>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "include/common/utils/python_adapter.h"
#include "kernel/akg/akg_kernel_json_generator.h"
#include "common/graph_kernel/split_umonad.h"
#include "common/graph_kernel/substitute_dropout.h"
#include "common/graph_kernel/graph_kernel_helper.h"
#include "common/graph_kernel/graph_kernel_flags.h"
#include "common/graph_kernel/adapter/callback_impl.h"
#include "common/graph_kernel/core/graph_kernel_utils.h"
#include "common/graph_kernel/core/convert_op_input_attr.h"
#include "backend/common/pass/inplace_assign_for_custom_op.h"
#include "kernel/common_utils.h"
#include "utils/ms_context.h"
#include "include/common/debug/anf_ir_dump.h"
#include "ir/func_graph_cloner.h"
#include "mindspore/core/ops/op_name.h"

namespace mindspore::graphkernel {
ExpanderPtr GetExpander(const AnfNodePtr &node, bool abstract) {
  ExpanderPtr expander =
    abstract
      ? std::make_shared<PyExpander>(std::static_pointer_cast<Callback>(std::make_shared<CallbackImplWithInferShape>()))
      : std::make_shared<PyExpander>(Callback::Instance());
  if (IsComplexOp(node)) {
    return ComplexOpDecorator::Creator(expander);
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
    {prim::kPrimVmapUnstackAssign->name(), {AttrToInputDeco::Creator}},
  };
  const auto iter = creators.find(GetCNodePrimitive(node)->name());
  if (iter != creators.end()) {
    expander = WrapExpander(expander, iter->second);
  }
  if (common::AnfAlgo::IsDynamicShape(node)) {
    expander = SetDynamicShapeAttrDeco::Creator(expander);
  }
  return expander;
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
    {kAllTarget, OpLevel_0, prim::kPrimVmapStackAssign},
    {kAllTarget, OpLevel_0, prim::kPrimVmapUnstackAssign},
    {kAllTarget, OpLevel_0, prim::kPrimSiLU},
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

AnfNodePtr TryExpandCNode(const AnfNodePtr &node, const std::function<bool(const CNodePtr &kernel_node)> &func) {
  if (!CanExpandFallback(node)) {
    return nullptr;
  }
  auto expander = GetExpander(node);
  expander = AttrToInputDeco::Creator(expander);
  auto res = expander->Run(node);
  auto expand_fg = GetCNodeFuncGraph(res);
  if (expand_fg != nullptr) {
    auto todos = TopoSort(expand_fg->get_return());
    for (const auto &n : todos) {
      auto cnode = n->cast<CNodePtr>();
      if (cnode == nullptr || !AnfUtils::IsRealKernel(cnode)) {
        continue;
      }
      auto suc = func(cnode);
      if (!suc) {
        MS_LOG(DEBUG) << "Expanding core ops [" << cnode->fullname_with_scope() << "] failed.";
        res = nullptr;
        break;
      }
    }
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kFully)) {
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

AnfNodePtr AttrToInputDeco::Run(const AnfNodePtr &node) {
  auto new_node = decorated_->Run(node);
  if (new_node == nullptr) {
    return nullptr;
  }
  auto new_cnode = dyn_cast<CNode>(new_node);
  auto expand_fg = GetCNodeFuncGraph(new_cnode);
  auto todos = TopoSort(expand_fg->get_return());
  for (const auto &node : todos) {
    ConvertOpUtils::ConvertAttrToInput(node);
  }
  new_cnode->set_input(0, NewValueNode(expand_fg));
  return new_cnode;
}

bool PyExpander::CreateJsonInfo(const AnfNodePtr &node, nlohmann::json *kernel_json) {
  DumpOption dump_option;
  dump_option.extract_opinfo_from_anfnode = true;
  AkgKernelJsonGenerator json_generator(dump_option, cb_);
  return json_generator.CollectJson(node, kernel_json);
}

FuncGraphPtr PyExpander::ExpandToGraphByCallPyFn(const CNodePtr &node) {
  MS_LOG(DEBUG) << "CallPyFn: [" << kGetGraphKernelExpanderOpList << "].";
  auto res = python_adapter::CallPyFn(kGraphKernelModule, kGetGraphKernelExpanderOpList);
  // parse result.
  if (py::isinstance<py::none>(res)) {
    MS_LOG(ERROR) << "CallPyFn: [" << kGetGraphKernelExpanderOpList << "] failed.";
    return nullptr;
  }

  std::string expander_op_list = py::cast<std::string>(res);
  auto op_name = AnfUtils::GetCNodeName(node);
  if (expander_op_list.find(op_name) == std::string::npos) {
    MS_LOG(DEBUG) << "Do not support to expand: " << op_name;
    return nullptr;
  }

  nlohmann::json kernel_json;
  if (!CreateJsonInfo(node, &kernel_json)) {
    constexpr int recursive_level = 2;
    MS_LOG(ERROR) << "Expand json info to: " << node->DebugString(recursive_level) << " failed, ori_json:\n"
                  << kernel_json.dump();
    return nullptr;
  }
  auto node_desc_str = kernel_json.dump();
  // call graph kernel ops generator.
  MS_LOG(DEBUG) << "CallPyFn: [" << kGetGraphKernelOpExpander << "] with input json:\n" << node_desc_str;
  auto ret = python_adapter::CallPyFn(kGraphKernelModule, kGetGraphKernelOpExpander, node_desc_str);
  // parse result.
  if (py::isinstance<py::none>(ret)) {
    MS_LOG(ERROR) << "CallPyFn: [" << kGetGraphKernelOpExpander << "] return invalid result, input json:\n"
                  << node_desc_str;
    return nullptr;
  }
  std::string kernel_desc_str = py::cast<std::string>(ret);
  if (kernel_desc_str.empty()) {
    return nullptr;
  }
  // decode json to func_graph.
  return JsonDescToAnf(kernel_desc_str);
}

FuncGraphPtr PyExpander::ExpandToGraph(const CNodePtr &node) {
  auto op_name = AnfUtils::GetCNodeName(node);
  // use cpp OpDesc in priority
  auto use_py = common::GetEnv("MS_DEV_PYEXPANDER");
  if (use_py.empty()) {
    if (expanders::OpDescFactory::Instance().HasOp(op_name)) {
      return DefaultExpander::ExpandToGraph(node);
    }
  }
  auto ms_context = MsContext::GetInstance();
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  if (pynative_mode && PyGILState_Check() == 0) {
    // Acquire Python GIL
    py::gil_scoped_acquire gil;
    if (PyGILState_Check() == 0) {
      MS_LOG(ERROR) << "Can not acquire python GIL.";
      return nullptr;
    }
    auto fg = ExpandToGraphByCallPyFn(node);
    py::gil_scoped_release rel;
    return fg;
  }
  return ExpandToGraphByCallPyFn(node);
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
  auto out = InlineClone(expanded_graph, main_graph, inp, cnode->input(0)->scope());
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
