/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/grad/ir/ir_grad.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "frontend/expander/bprop/bprop.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/profiler.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/pynative/grad/jit/jit_call_graph.h"
#include "pipeline/pynative/pynative_utils.h"
#include "utils/info.h"
#include "utils/profile.h"

namespace mindspore {
namespace pynative {
namespace autograd {
namespace {
void SetJitCallGraph(const CNodePtr &cnode, const FuncGraphPtr &call_graph, const std::string &cache_key,
                     const GraphCallCondition &graph_call_condition) {
  MS_EXCEPTION_IF_NULL(cnode);
  common::AnfAlgo::SetNodeAttr(kAttrJitCallNode, MakeValue(true), cnode);
  auto graph_call_back = PyNativeAlgo::AutoGrad::CreateGraphCallBack(call_graph, cache_key, graph_call_condition);
  cnode->set_user_data<JitCallGraph>(std::make_shared<JitCallGraph>(graph_call_back));
}

bool IsOutputBothEmpty(const AnfNodePtr &inputs_grad, const AnfNodePtr &weights_grad) {
  if (!inputs_grad->isa<CNode>() || !weights_grad->isa<CNode>()) {
    return false;
  }
  auto inputs_grad_cnode = inputs_grad->cast<CNodePtr>();
  auto weights_grad_cnode = weights_grad->cast<CNodePtr>();
  if (!IsPrimitiveCNode(inputs_grad_cnode, prim::kPrimMakeTuple) ||
      !IsPrimitiveCNode(weights_grad_cnode, prim::kPrimMakeTuple)) {
    return false;
  }
  constexpr int kEmptyTupeSize = 1;
  if (inputs_grad_cnode->size() != kEmptyTupeSize || weights_grad_cnode->size() != kEmptyTupeSize) {
    return false;
  }
  return true;
}

AnfNodePtr GenerateEmptyTupleValue() {
  std::vector<ValuePtr> value_list;
  auto inputs_value = std::make_shared<ValueTuple>(value_list);
  auto weights_value = std::make_shared<ValueTuple>(value_list);
  std::vector<ValuePtr> tuple_list{inputs_value, weights_value};
  auto tuple_value = std::make_shared<ValueTuple>(tuple_list);
  return PyNativeAlgo::Common::CreateValueNodeByValue(tuple_value);
}

bool IsValidTensorInput(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return abs->isa<abstract::AbstractTensor>() || abs->isa<abstract::AbstractSparseTensor>();
}

AnfNodePtr GetTupleItemNodeInput(const KernelGraphPtr &tape, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(tape);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfNodePtr new_cnode = nullptr;
  if (IsPrimitive(cnode->input(kIndex1), prim::kPrimTupleGetItem)) {
    auto inner_cnode = cnode->input(kIndex1)->cast<CNodePtr>();
    new_cnode = tape->FuncGraph::NewCNode(
      {inner_cnode->input(kIndex0), GetTupleItemNodeInput(tape, inner_cnode), inner_cnode->input(kIndex2)});
  } else {
    AnfNodePtrList new_inputs{cnode->inputs().begin(), cnode->inputs().end()};
    new_cnode = tape->FuncGraph::NewCNode(new_inputs);
  }
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  return new_cnode;
}

bool IsConstant(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    const auto &tensor = value->cast<tensor::BaseTensorPtr>();
    auto auto_grad_meta_data = tensor->auto_grad_meta_data();
    if (auto_grad_meta_data == nullptr) {
      return true;
    }
    if (auto_grad_meta_data->input_type() == InputType::kParameter ||
        auto_grad_meta_data->input_type() == InputType::kInput) {
      return false;
    }
    auto k_node = auto_grad_meta_data->k_node();
    if (k_node != nullptr) {
      return false;
    }
    return true;
  } else if (value->isa<ValueSequence>()) {
    auto val_seq = value->cast<ValueSequencePtr>();
    return std::all_of(val_seq->value().begin(), val_seq->value().end(),
                       [](const ValuePtr &value) { return IsConstant(value); });
  } else if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    return IsConstant(coo_tensor->GetIndices());
  } else if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    return IsConstant(csr_tensor->GetIndices());
  }
  return true;
}
}  // namespace

AnfNodePtr IrFunctionNode::HyperAdd(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);

  if (PyNativeAlgo::AutoGrad::IsZerosLikeNode(left_node)) {
    return right_node;
  }
  if (PyNativeAlgo::AutoGrad::IsZerosLikeNode(right_node)) {
    return left_node;
  }
  if (!IsPrimitiveCNode(left_node, prim::kPrimMakeTuple)) {
    auto add_result = tape_->FuncGraph::NewCNode({NewValueNode(prim::kPrimAdd), left_node, right_node});
    add_result->set_abstract(right_node->abstract());
    return add_result;
  }
  if (IsPrimitiveCNode(left_node, prim::kPrimMakeTuple) && IsPrimitiveCNode(right_node, prim::kPrimMakeTuple)) {
    auto left_cnode = left_node->cast<CNodePtr>();
    auto right_cnode = right_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(right_cnode);
    AnfNodePtrList inputs = {NewValueNode(prim::kPrimMakeTuple)};
    AbstractBasePtrList abs;
    for (size_t i = 1; i < left_cnode->size(); ++i) {
      auto add_result = HyperAdd(left_cnode->input(i), right_cnode->input(i));
      (void)abs.emplace_back(add_result->abstract());
      (void)inputs.emplace_back(add_result);
    }
    auto add_tuple = tape_->FuncGraph::NewCNode(inputs);
    add_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abs));
    return add_tuple;
  }
  MS_LOG(EXCEPTION) << "Unknown cnode type" << left_node->DebugString();
}

void IrFunctionNode::AddNextEdge(const VariablePtr &next_variable, const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(next_variable);
  MS_EXCEPTION_IF_NULL(din);
  // next_node and its corresponding din
  (void)next_edges_.emplace_back(next_variable, din);
  if (din == fake_dout_) {
    (void)need_replace_edges_.emplace_back(next_edges_.size() - 1);
  }
}

void IrFunctionNode::UpdateAccumulativeDout(const AnfNodePtr &new_dout) {
  MS_EXCEPTION_IF_NULL(new_dout);
  accumulate_dout_ = HyperAdd(accumulate_dout_, new_dout);
}

void IrFunctionNode::ReplaceEdges() {
  MS_EXCEPTION_IF_NULL(accumulate_dout_);
  for (const auto index : need_replace_edges_) {
    next_edges_[index].second = accumulate_dout_;
  }
}

IrGrad::IrGrad(const std::vector<ValuePtr> &input_param_values, const AbstractBasePtrList &abs_list,
               size_t op_num_in_bprop_graph, bool grad_by_value, bool is_run_recompute)
    : ad_param_(std::make_shared<AdParam>()) {
  ad_param()->tape_->debug_info()->set_name("grad_top");
  MS_LOG(DEBUG) << "Start IrGrad, input size: " << input_param_values.size();
  ad_param()->variable_adjoint_set_.reserve(op_num_in_bprop_graph);
  ad_param()->anfnode_to_variable_adjoint_.reserve(op_num_in_bprop_graph);
  ad_param()->users_.dout_user_.reserve(op_num_in_bprop_graph);
  ad_param()->weights_used_in_graph_.reserve(op_num_in_bprop_graph);
  param_meta_grad_info_.reserve(op_num_in_bprop_graph);

  for (size_t i = 0; i < input_param_values.size(); ++i) {
    auto input_parameter = ad_param()->fg_->add_parameter();
    input_parameter->set_abstract(abs_list[i]);
    input_parameter->set_name(input_parameter->UniqueName());
    TraceGuard trace_guard(std::make_shared<TraceCopy>(input_parameter->debug_info()));
    auto tape_parameter = ad_param()->tape_->add_parameter();
    tape_parameter->set_abstract(abs_list[i]);

    auto zeros_like_dout = PyNativeAlgo::AutoGrad::BuildSpecialNode(
      ad_param()->tape_, PyNativeAlgo::AutoGrad::GetFakeZeroTensor(), abs_list[i], SpecialType::kZerosLikeType);
    auto func_node = std::make_shared<IrFunctionNode>(ad_param()->tape_, zeros_like_dout);
    auto input_adjoint = std::make_shared<IrVariable>(func_node, input_param_values[i], true);

    if (!input_param_values[i]->isa<ValueSequence>()) {
      PyNativeAlgo::AutoGrad::SetGradInfoForInputs(input_param_values[i], input_adjoint, &param_meta_grad_info_,
                                                   input_parameter);
    } else {
      input_adjoint->set_is_need_grad(false);
    }
    (void)cell_inputs_.emplace_back(input_parameter, input_adjoint);
    (void)ad_param()->variable_adjoint_set_.insert(input_adjoint);
  }

  grad_by_value_ = grad_by_value;
  device_target_ = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  ir_bprop_ = std::make_unique<IrBprop>(ad_param_, device_target_, grad_by_value_, is_run_recompute);
}

bool IrGrad::KPynativeOp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);

  auto &prim = grad_param->op_grad_info->op_prim;
  if (!PyNativeAlgo::AutoGrad::IsPrimNeedGrad(prim) ||
      (grad_by_value_ && !PyNativeAlgo::AutoGrad::NeedGrad(grad_param->op_grad_info->input_value))) {
    MS_LOG(DEBUG) << "Prim " << prim->name() << " does not need to do op grad.";
    return true;
  }

  auto cloned_value = grad_param->op_grad_info->out_value;
  if (grad_param->op_grad_info->out_value->isa<ValueSequence>()) {
    cloned_value = ShallowCopyTensorValue(grad_param->op_grad_info->out_value);
    PyNativeAlgo::Common::ClearDeviceAddress(cloned_value);
  }

  PyNativeAlgo::AutoGrad::CheckAndSetAbstract(grad_param->op_grad_info);
  // construct zeroslike placeholder, if need use in bprop, we replace it in backprogate.
  AnfNodePtr dout =
    PyNativeAlgo::AutoGrad::BuildSpecialNode(ad_param()->tape_, PyNativeAlgo::AutoGrad::GetFakeZeroTensor(),
                                             grad_param->op_grad_info->out_abs, SpecialType::kZerosLikeType);
  auto fn = std::make_shared<IrFunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<IrVariable>(fn, cloned_value);
  // Custom forward cnode no need record in bprop graph, because it is a flag cnode for run python. So just create
  // bprop_cut grad op is ok
  bool is_custom_prim =
    IsPrimitiveEquals(prim, prim::kPrimHookBackward) || IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook);
  AnfNodePtr k_node = nullptr;
  if (!grad_by_value_ && !is_custom_prim) {
    k_node = BuildKNode(NewValueNode(prim), grad_param, true);
    SetKNodeInfo(grad_param->op_grad_info->out_value, k_node, grad_param->op_grad_info->out_abs);
    need_do_manager_replace_ = true;
  }
  CNodePtr input_node = ConstructBpropGraphInput(grad_param, dout, variable_adjoint, k_node, is_custom_prim);
  MS_LOG(DEBUG) << "Construct input cnode: " << input_node->DebugString();
  // Gradient outputs
  std::vector<CNodePtr> outputs;
  if (!is_custom_prim) {
    auto ret = BpropExpander(&outputs, &ad_param()->users_).Run(input_node, grad_param->op_grad_info->input_value);
    // cppcheck-suppress unreadVariable
    if (MS_UNLIKELY(!ret || outputs.empty())) {
      MS_LOG(DEBUG) << "Expander has no bprop of this prim: " << prim->name();
      ir_bprop_->BuildCustomBpropCNode(input_node, prim, &outputs);
    }
  } else {
    PyNativeAlgo::AutoGrad::CheckRecomputeInputs(grad_param);
    ir_bprop_->BuildBPropCutCNode(input_node, prim, &outputs, grad_param->op_grad_info->weight_size,
                                  grad_param->op_grad_info->is_need_recompute);
  }
  // cppcheck-suppress unreadVariable
  if (MS_UNLIKELY(outputs.empty())) {
    MS_LOG(DEBUG) << "This op has not custom bprop: " << prim->name();
    PyNativeAlgo::AutoGrad::BuildFakeBpropCNode(input_node, &outputs);
    variable_adjoint->set_is_fake_bprop(true);
    variable_adjoint->set_fake_prim_name(prim->name());
  }
  (void)ad_param()->variable_adjoint_set_.insert(variable_adjoint);
  PyNativeAlgo::AutoGrad::SetGradMetaData(grad_param->op_grad_info->out_value, variable_adjoint);
  ir_bprop_->UpdateNextEdges(variable_adjoint, outputs, grad_param->op_grad_info->input_value,
                             grad_param->op_grad_info->input_abs, prim->name());
  return true;
}

bool IrGrad::KPynativeWithFProp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  MS_LOG(DEBUG) << "Do KPynativeWithFProp";
  AnfNodePtrList args_node_list;
  CNodePtr bprop_cnode = nullptr;
  AnfNodePtr k_node = nullptr;
  AnfNodePtr dout = nullptr;
  if (grad_by_value_) {
    for (size_t i = 0; i < grad_param->input_size; ++i) {
      if (PyNativeAlgo::Common::IsParam(grad_param->op_grad_info->input_value_grad_type[i])) {
        auto parameter = ir_bprop_->MapParameter(grad_param->op_grad_info->input_value[i],
                                                 grad_param->op_grad_info->input_abs[i], &param_meta_grad_info_);
        MS_EXCEPTION_IF_NULL(parameter);
        (void)args_node_list.emplace_back(parameter);
        continue;
      }
      // Valuenode, node
      const auto value_node = PyNativeAlgo::Common::CreateValueNodeByValue(
        grad_param->op_grad_info->input_value[i], grad_param->op_grad_info->input_abs[i]->Clone());
      auto cnode = PyNativeAlgo::Common::ConvertValueSequenceToMakeTuple(value_node, ad_param()->tape_);
      (void)args_node_list.emplace_back(cnode);
    }
    bprop_cnode = GetBpropGraphCNode(grad_param, args_node_list, &dout);
  } else {
    k_node = BuildKNode(NewValueNode(grad_param->source_fg), grad_param, false);
    BuildKNodeListForHighOrderGraph(grad_param->op_grad_info->input_value, grad_param->op_grad_info->input_abs,
                                    &args_node_list);
    bprop_cnode = GetBpropGraphCNode(grad_param, args_node_list, &dout);
  }
  auto fn = std::make_shared<IrFunctionNode>(ad_param()->tape_, dout);
  auto variable_adjoint = std::make_shared<IrVariable>(fn, grad_param->op_grad_info->out_value);
  variable_adjoint->set_k_node(k_node);
  std::vector<CNodePtr> outputs;
  for (size_t i = 0; i < grad_param->input_size; ++i) {
    CNodePtr din = ad_param()->tape_->FuncGraph::NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), bprop_cnode, NewValueNode(SizeToLong(i))});
    din->set_abstract(grad_param->op_grad_info->input_abs[i]);
    (void)outputs.emplace_back(din);
  }
  ir_bprop_->UpdateNextEdges(variable_adjoint, outputs, grad_param->op_grad_info->input_value,
                             grad_param->op_grad_info->input_abs);
  (void)ad_param()->variable_adjoint_set_.insert(variable_adjoint);
  (void)ad_param()->anfnode_to_variable_adjoint_.insert(std::make_pair(grad_param->cnode, variable_adjoint));
  PyNativeAlgo::AutoGrad::SetGradMetaData(grad_param->op_grad_info->out_value, variable_adjoint);
  SetKNodeInfo(grad_param->op_grad_info->out_value, k_node, grad_param->op_grad_info->out_abs);
  return true;
}

CNodePtr IrGrad::GetBPropCNode(const GradParamPtr &grad_param, const AnfNodePtrList &args,
                               const FuncGraphPtr &bprop_graph, bool cache_hit, AnfNodePtr *const tape_dout) {
  AnfNodePtrList bprop_inputs(args.begin(), args.end());
  bool is_jit_dynamic_shape = grad_param->is_jit_graph && grad_param->use_dynamic_shape_process;
  // Save replace info in first time
  if (!cache_hit && is_jit_dynamic_shape && grad_param->has_added_v) {
    const auto &jit = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor()->jit();
    jit->SaveForwardOutputTensorInfoInBpropGraph(bprop_graph);
  }

  // Call by tape_
  MS_EXCEPTION_IF_NULL(tape_dout);
  *tape_dout = PyNativeAlgo::AutoGrad::BuildSpecialNode(ad_param()->tape_, PyNativeAlgo::AutoGrad::GetFakeZeroTensor(),
                                                        grad_param->op_grad_info->out_abs, SpecialType::kZerosLikeType);
  if (is_jit_dynamic_shape && grad_param->op_grad_info->out_abs->isa<abstract::AbstractSequence>()) {
    auto abs_seq = grad_param->op_grad_info->out_abs->cast<abstract::AbstractSequencePtr>();
    // Dynamic len has no size current
    if (!abs_seq->dynamic_len()) {
      for (size_t i = 0; i < abs_seq->size(); ++i) {
        CNodePtr din = ad_param()->tape_->FuncGraph::NewCNode(
          {NewValueNode(prim::kPrimTupleGetItem), *tape_dout, NewValueNode(SizeToLong(i))});
        din->set_abstract(abs_seq->elements()[i]);
        (void)bprop_inputs.emplace_back(din);
        ir_bprop_->AddUser(*tape_dout, din, kIndex1);
      }
    }
  } else {
    (void)bprop_inputs.emplace_back(*tape_dout);
  }
  (void)bprop_inputs.insert(bprop_inputs.cbegin(), NewValueNode(bprop_graph));
  // get_bprop is a call node
  auto bprop_cnode = ad_param()->tape_->FuncGraph::NewCNode(bprop_inputs);
  bprop_cnode->set_abstract(bprop_graph->output()->abstract());
  if (is_jit_dynamic_shape) {
    GraphCallCondition graph_call_condition{grad_param->is_control_flow, grad_param->is_jit_graph,
                                            grad_param->use_dynamic_shape_process, false, false};
    SetJitCallGraph(bprop_cnode, bprop_graph, grad_param->graph_cache_key, graph_call_condition);
    ad_param()->tape_->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  }
  // For replacing parameter and dout.
  for (size_t i = 1; i < bprop_inputs.size(); ++i) {
    ir_bprop_->AddUser(bprop_inputs[i], bprop_cnode, i);
  }
  return bprop_cnode;
}

CNodePtr IrGrad::GetBpropGraphCNode(const GradParamPtr &grad_param, const AnfNodePtrList &args,
                                    AnfNodePtr *const tape_dout) {
  MS_EXCEPTION_IF_NULL(grad_param);
  auto [cache_hit, bprop_graph] = ir_bprop_->GetBpropGraph(grad_param);
  if (grad_param->is_control_flow || grad_param->is_jit_self_dynamic_shape) {
    need_do_manager_replace_ = true;
  }
  return GetBPropCNode(grad_param, args, bprop_graph, cache_hit, tape_dout);
}

void IrGrad::UpdateOutputNodeOfTopCell(const ValuePtr &sens_out) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                     runtime::ProfilerEvent::kPyNativeGradUpdateSens,
                                     runtime::ProfilerRecorder::kNoName, true);
  MS_EXCEPTION_IF_NULL(sens_out);
  MS_LOG(DEBUG) << "Real output of top cell is " << PyNativeAlgo::Common::GetIdByValue(sens_out);
  ad_param()->sens_value_ = sens_out;
  UpdateSensParameter(ad_param()->sens_value_);
}

FuncGraphPtr IrGrad::Finish(const tensor::BaseTensorPtrList &weights, const std::vector<size_t> &grad_position,
                            const GradAttr &grad_attr) {
  // Set sens node and weights node
  SetSensAndWeights(weights, grad_attr.has_sens);

  // BackPropagate sensitivity, except when the last node is a valuenode which may be obtained by constant folding;
  if (ad_param()->last_variable_->is_need_grad() && !ad_param()->last_variable_->is_leaf()) {
    ir_bprop_->BackPropagate();
  }
  SetOutput(weights, grad_position, grad_attr);
  // Replace Parameter of primal func graph with parameter of ad_param()->tape_;
  ReplacePrimalParameter(grad_attr.has_sens);
  PyNativeAlgo::Common::DumpGraphIR("before_final_opt.ir", ad_param()->tape_);
  return ad_param()->tape_;
}

CNodePtr IrGrad::ConstructBpropGraphInput(const GradParamPtr &grad_param, const AnfNodePtr &dout,
                                          const VariablePtr &variable_adjoint, const AnfNodePtr &k_node,
                                          bool is_custom_prim) {
  MS_EXCEPTION_IF_NULL(grad_param);
  AnfNodePtrList node_list;
  (void)node_list.emplace_back(NewValueNode(grad_param->op_grad_info->op_prim));
  if (grad_by_value_ || is_custom_prim) {
    // If recompute, we do not push weight data to cnode inputs.
    for (size_t i = 0; i < grad_param->input_size; ++i) {
      if (PyNativeAlgo::Common::IsParam(grad_param->op_grad_info->input_value_grad_type[i])) {
        // To solve the input is a tuple like (parameter, ...)
        auto parameter = ir_bprop_->MapParameter(grad_param->op_grad_info->input_value[i],
                                                 grad_param->op_grad_info->input_abs[i], &param_meta_grad_info_);
        MS_EXCEPTION_IF_NULL(parameter);
        (void)node_list.emplace_back(parameter);
        continue;
      }
      // Node abstract obj may free, so v node abstract will be not correct
      (void)node_list.emplace_back(PyNativeAlgo::Common::CreateValueNodeByValue(
        grad_param->op_grad_info->input_value[i], grad_param->op_grad_info->input_abs[i]->Clone()));
    }
    // Hook run by single op
    if (!ir_bprop_->bprop_graph_run_by_single_op()) {
      ir_bprop()->set_bprop_graph_run_by_single_op([&grad_param]() {
        auto tensor = grad_param->op_grad_info->out_value->template cast<tensor::BaseTensorPtr>();
        if (tensor == nullptr || tensor->auto_grad_meta_data() == nullptr) {
          return false;
        }
        auto auto_grad_meta = tensor->auto_grad_meta_data();
        return auto_grad_meta->is_register_hook();
      }());
    }
    // Set out
    (void)node_list.emplace_back(PyNativeAlgo::Common::CreateValueNodeByValue(grad_param->op_grad_info->out_value,
                                                                              grad_param->op_grad_info->out_abs));
  } else {
    // Input is a Parameter or cnode, not a value node
    BuildKNodeListFromPrimalCNode(grad_param->op_grad_info->input_value, grad_param->op_grad_info->input_abs,
                                  &node_list);
    // Set out
    MS_EXCEPTION_IF_NULL(variable_adjoint);
    (void)node_list.emplace_back(k_node);
  }
  // Set dout
  (void)node_list.emplace_back(dout);
  auto input_node = ad_param()->tape_->FuncGraph::NewCNode(node_list);
  return input_node;
}

void IrGrad::BuildKNodeListFromPrimalCNode(const ValuePtrList &input_value,
                                           const abstract::AbstractBasePtrList &input_abs,
                                           AnfNodePtrList *const node_list) {
  for (size_t i = 0; i < input_value.size(); ++i) {
    (void)node_list->emplace_back(BuildKNodeForCNodeInput(input_value[i], input_abs[i]));
    MS_LOG(DEBUG) << "Get knode for input:  " << PyNativeAlgo::Common::GetIdByValue(input_value[i]);
  }
}

AnfNodePtr IrGrad::BuildKNodeForCNodeInput(const ValuePtr &input, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(input);
  if (input->isa<tensor::BaseTensor>()) {
    const auto &tensor = input->cast<tensor::BaseTensorPtr>();
    const auto &auto_grad_meta_data = tensor->auto_grad_meta_data();
    if (auto_grad_meta_data != nullptr) {
      auto k_node = auto_grad_meta_data->k_node();
      if (k_node != nullptr) {
        return k_node;
      }
      if (PyNativeAlgo::Common::IsParam(auto_grad_meta_data->input_type())) {
        return ir_bprop_->MapParameter(input, abs, &param_meta_grad_info_);
      }
    }
  } else if (input->isa<ValueSequence>() && !IsConstant(input)) {
    AnfNodePtrList inputs;
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    const auto &val_sequence = input->cast<ValueSequencePtr>()->value();
    const auto &abs_sequence = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_sequence);
    if (val_sequence.size() != abs_sequence->size()) {
      MS_LOG(EXCEPTION) << "Get value sequence size " << val_sequence.size() << " not equal to abstract size "
                        << abs_sequence->size();
    }
    for (size_t i = 0; i < val_sequence.size(); ++i) {
      (void)inputs.emplace_back(BuildKNodeForCNodeInput(val_sequence[i], abs_sequence->elements()[i]));
    }
    auto k_node = ad_param_->tape_->FuncGraph::NewCNode(inputs);
    k_node->set_abstract(abs);
    return k_node;
  }
  auto value_node = NewValueNode(input);
  value_node->set_abstract(abs);
  return value_node;
}

void IrGrad::BuildKNodeListForHighOrderGraph(const ValuePtrList &input_value,
                                             const abstract::AbstractBasePtrList &input_abs,
                                             AnfNodePtrList *const node_list) {
  for (size_t i = 0; i < input_value.size(); ++i) {
    const auto knode = BuildKNodeForCNodeInput(input_value[i], input_abs[i]);
    // Convert value sequence to make tuple, so that finalpass can eliminate tuplegetitem.
    // BuildKnodeForTuplgeGetItem now do not support input is valuesequence.
    if (knode->isa<ValueNode>()) {
      auto value_node = knode->cast<ValueNodePtr>();
      (void)node_list->emplace_back(
        PyNativeAlgo::Common::ConvertValueSequenceToMakeTuple(value_node, ad_param()->tape_));
    } else {
      (void)node_list->emplace_back(knode);
    }

    MS_LOG(DEBUG) << "Get knode for input:  " << PyNativeAlgo::Common::GetIdByValue(input_value[i]);
  }
}

void IrGrad::SetKNodeInfo(const ValuePtr &value, const AnfNodePtr &k_node, const AbstractBasePtr &out_abs) {
  if (value->isa<tensor::BaseTensor>()) {
    auto tensor = value->cast<tensor::BaseTensorPtr>();
    auto auto_grad_meta_data = tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto_grad_meta_data->set_k_node(k_node);
    (void)k_nodes_used_in_graph_.emplace_back(k_node);
  } else if (value->isa<ValueSequence>()) {
    const auto &value_sequence = value->cast<ValueSequencePtr>()->value();
    const auto &abs_seq = out_abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (abs_seq->dynamic_len()) {
      return;
    }
    if (value_sequence.size() != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Get value sequence size " << value_sequence.size() << " not equal to abstract size "
                        << abs_seq->size();
    }
    for (size_t i = 0; i < value_sequence.size(); ++i) {
      auto sub_k_node = ad_param()->tape_->FuncGraph::NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), k_node, NewValueNode(static_cast<int64_t>(i))});
      sub_k_node->set_abstract(abs_seq->elements()[i]);
      SetKNodeInfo(value_sequence[i], sub_k_node, abs_seq->elements()[i]);
    }
  }
}

AnfNodePtr IrGrad::BuildKNode(const AnfNodePtr &prim, const GradParamPtr &grad_param, bool from_single_op) {
  MS_EXCEPTION_IF_NULL(grad_param);
  AnfNodePtrList node_list;
  (void)node_list.emplace_back(prim);
  for (size_t i = 0; i < grad_param->input_size; ++i) {
    (void)node_list.emplace_back(
      BuildKNodeForCNodeInput(grad_param->op_grad_info->input_value[i], grad_param->op_grad_info->input_abs[i]));
  }
  auto k_node = ad_param()->tape_->FuncGraph::NewCNode(node_list);
  k_node->set_abstract(grad_param->op_grad_info->out_abs);
  k_node->AddAttr(bprop_pass::kIsKNode, MakeValue(true));
  if (from_single_op && grad_param->out_used_in_bporp_graph) {
    auto v_node = PyNativeAlgo::Common::CreateValueNodeByValue(grad_param->op_grad_info->out_value,
                                                               grad_param->op_grad_info->out_abs);
    k_node->set_forward(v_node, "");
    ad_param()->tape_->set_used_forward_nodes({k_node});
  }
  MS_LOG(DEBUG) << "Build knode " << k_node->DebugString();
  return k_node;
}

void IrGrad::UpdateSensParameter(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    const auto &sens_tensor = value->cast<tensor::BaseTensorPtr>();
    const auto &auto_grad_meta_data = sens_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    const auto variable = auto_grad_meta_data->variable();
    // Return input parameter or weight parameter for net, if v is parameter just entry once
    if (auto_grad_meta_data->input_type() == InputType::kParameter && variable == nullptr) {
      (void)ir_bprop_->AddParameterNode(sens_tensor,
                                        PyNativeAlgo::Common::SetAbstractValueToAnyValue(sens_tensor->ToAbstract()));
      param_meta_grad_info_.emplace_back(sens_tensor, auto_grad_meta_data);
    }
  } else if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>()->value();
    for (const auto &v : value_seq) {
      UpdateSensParameter(v);
    }
  } else if (value->isa<ValueDictionary>()) {
    auto dic_v = value->cast<ValueDictionaryPtr>();
    for (const auto &v : dic_v->value()) {
      UpdateSensParameter(v.second);
    }
  }
}

ParameterPtr IrGrad::ExtractParameter(const tensor::BaseTensorPtr &tensor) const {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &auto_grad_meta_data = tensor->auto_grad_meta_data();
  if (auto_grad_meta_data != nullptr && PyNativeAlgo::Common::IsParam(auto_grad_meta_data->input_type())) {
    return auto_grad_meta_data->parameter();
  }
  return nullptr;
}

void IrGrad::SetSensAndWeights(const tensor::BaseTensorPtrList &weights, bool has_sens_arg) {
  const auto &sens_abstract = ir_bprop_->BuildForwardLastNode();
  ParameterPtr sens_param = nullptr;
  if (has_sens_arg) {
    sens_param = ad_param()->tape_->add_parameter();
    sens_param->set_name(sens_param->UniqueName());
    sens_param->debug_info()->set_name("sens");
    sens_param->set_abstract(sens_abstract);
  }
  // Update dout for dout
  MS_EXCEPTION_IF_NULL(ad_param()->last_variable_);
  if (ad_param()->last_variable_->is_need_grad()) {
    if (has_sens_arg) {
      ad_param()->last_variable_->ir_function_node()->UpdateAccumulativeDout(sens_param);
    } else {
      ad_param()->last_variable_->ir_function_node()->UpdateAccumulativeDout(PyNativeAlgo::AutoGrad::BuildSpecialNode(
        ad_param()->tape_, ad_param()->sens_value_, sens_abstract, SpecialType::kOnesLikeType));
    }
  }
  // Add weights parameter
  need_grad_weights_.reserve(weights.size());
  for (const auto &weight_tensor : weights) {
    (void)need_grad_weights_.emplace(weight_tensor->id());
    UpdateTapeParameter(weight_tensor);
  }
  for (auto &weight : ad_param_->weights_used_in_graph_) {
    auto tensor = PyNativeAlgo::Common::GetTensorFromParam(weight);
    MS_EXCEPTION_IF_NULL(tensor);
    // Need get grad, but not used in bprop graph
    if (need_grad_weights_.find(tensor->id()) == need_grad_weights_.end()) {
      UpdateTapeParameter(tensor);
    }
  }
}

AnfNodePtr IrGrad::GetGradNodeByIndex(const tensor::BaseTensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
  auto variable = auto_grad_meta_data->variable();
  MS_LOG(DEBUG) << "Get variable " << (variable != nullptr ? variable->ToString() : "is nullptr");
  if (variable != nullptr && variable->is_need_grad()) {
    // If weight used in the forward network, but requires_grad is false, return zero like.
    if (tensor->param_info() != nullptr && !tensor->param_info()->requires_grad()) {
      MS_LOG(INFO) << "weight participate in forward calculation, but requires_grad is false";
      return PyNativeAlgo::AutoGrad::BuildSpecialNode(ad_param()->tape_, tensor, nullptr, SpecialType::kZerosLikeType);
    }
    const auto &ir_variable = std::dynamic_pointer_cast<IrVariable>(variable);
    MS_EXCEPTION_IF_NULL(ir_variable);
    return ir_variable->RealDout();
  }
  MS_LOG(INFO) << "weight not participate in forward calculation, but requires grad, id: "
               << PyNativeAlgo::Common::GetIdByValue(tensor);
  return PyNativeAlgo::AutoGrad::BuildSpecialNode(ad_param()->tape_, tensor, nullptr, SpecialType::kZerosLikeType);
}

AnfNodePtr IrGrad::GetInputGrad(bool grad_all_inputs, bool get_by_position, const std::vector<size_t> &grad_position) {
  std::vector<size_t> grad_pos_list;
  if (get_by_position) {
    grad_pos_list = grad_position;
  } else if (grad_all_inputs) {
    grad_pos_list.resize(cell_inputs_.size());
    iota(grad_pos_list.begin(), grad_pos_list.end(), 0);
  } else {
    return nullptr;
  }

  AnfNodePtrList inputs_grad_list{NewValueNode(prim::kPrimMakeTuple)};
  AbstractBasePtrList inputs_grad_spec;
  if (!cell_inputs_.empty()) {
    for (size_t index : grad_pos_list) {
      if (index >= cell_inputs_.size()) {
        MS_LOG(EXCEPTION) << "Position index " << index << " is exceed input size.";
      }
      // Tuple, List, scalar will be ignored
      if (!IsValidTensorInput(cell_inputs_[index].first->abstract())) {
        MS_LOG(DEBUG) << "Get input node is not tensor "
                      << ", abs " << cell_inputs_[index].first->abstract()->ToString();
        continue;
      }
      auto ir_variable = std::dynamic_pointer_cast<IrVariable>(cell_inputs_[index].second);
      MS_EXCEPTION_IF_NULL(ir_variable);
      auto real_dout = ir_variable->RealDout();
      MS_EXCEPTION_IF_NULL(real_dout);
      (void)inputs_grad_list.emplace_back(real_dout);
      (void)inputs_grad_spec.emplace_back(real_dout->abstract());
    }
    constexpr size_t single_pos_size = 1;
    if (get_by_position && inputs_grad_spec.size() == single_pos_size) {
      // First elem is prim
      return inputs_grad_list[single_pos_size];
    }
  }
  auto input_grad_ret = ad_param()->tape_->FuncGraph::NewCNode(inputs_grad_list);
  input_grad_ret->set_abstract(std::make_shared<abstract::AbstractTuple>(inputs_grad_spec));
  return input_grad_ret;
}

AnfNodePtr IrGrad::GetWeightGrad(bool grad_weights, const tensor::BaseTensorPtrList &weights,
                                 bool weight_param_is_tuple) {
  // No need to return gradient of weights.
  if (!grad_weights) {
    return nullptr;
  }
  if (weight_param_is_tuple) {
    AnfNodePtrList weights_grad_list{NewValueNode(prim::kPrimMakeTuple)};
    AbstractBasePtrList weights_grad_spec;
    for (const auto &weight : weights) {
      auto grad_node = GetGradNodeByIndex(weight);
      MS_EXCEPTION_IF_NULL(grad_node);
      (void)weights_grad_list.emplace_back(grad_node);
      (void)weights_grad_spec.emplace_back(grad_node->abstract());
    }
    auto weight_grad_ret = ad_param()->tape_->FuncGraph::NewCNode(weights_grad_list);
    weight_grad_ret->set_abstract(std::make_shared<abstract::AbstractTuple>(weights_grad_spec));
    return weight_grad_ret;
  } else {
    return GetGradNodeByIndex(weights[0]);
  }
}

void IrGrad::SetOutput(const tensor::BaseTensorPtrList &weights, const std::vector<size_t> &grad_position,
                       const GradAttr &grad_attr) {
  auto inputs_grad_ret = GetInputGrad(grad_attr.grad_all_inputs, grad_attr.get_by_position, grad_position);
  auto weights_grad_ret = GetWeightGrad(grad_attr.grad_weights, weights, grad_attr.weight_param_is_tuple);
  // Gradients wrt inputs and weights.
  if (inputs_grad_ret != nullptr && weights_grad_ret != nullptr) {
    if (IsOutputBothEmpty(inputs_grad_ret, weights_grad_ret)) {
      auto tape_output = GenerateEmptyTupleValue();
      ad_param()->tape_->set_output(tape_output);
    } else {
      auto tape_output =
        ad_param()->tape_->FuncGraph::NewCNode({NewValueNode(prim::kPrimMakeTuple), inputs_grad_ret, weights_grad_ret});
      tape_output->set_abstract(std::make_shared<abstract::AbstractTuple>(
        abstract::AbstractBasePtrList{inputs_grad_ret->abstract(), weights_grad_ret->abstract()}));
      ad_param()->tape_->set_output(tape_output);
    }
    return;
  }
  // Gradients wrt inputs.
  if (inputs_grad_ret != nullptr) {
    ad_param()->tape_->set_output(inputs_grad_ret);
    return;
  }
  // Gradients wrt weights.
  if (weights_grad_ret != nullptr) {
    ad_param()->tape_->set_output(weights_grad_ret);
    return;
  }
  // grad_all_inputs, grad_weights and get_by_position are all false.
  AnfNodePtr tape_output = nullptr;
  if (cell_inputs_.empty()) {
    // If no input nodes, return empty tuple.
    tape_output = ad_param()->tape_->FuncGraph::NewCNode({NewValueNode(prim::kPrimMakeTuple)});
    abstract::AbstractBasePtrList abs{};
    tape_output->set_abstract(std::make_shared<abstract::AbstractTuple>(abs));
  } else {
    // If there are input nodes, return gradient of first input node.
    // Tuple, List, scalar will be ignore
    if (IsValidTensorInput(cell_inputs_[0].first->abstract())) {
      auto ir_variable = std::dynamic_pointer_cast<IrVariable>(cell_inputs_[kIndex0].second);
      MS_EXCEPTION_IF_NULL(ir_variable);
      tape_output = ir_variable->RealDout();
    } else {
      MS_LOG(DEBUG) << "Get first input node is not tensor " << cell_inputs_[0].second->out_value()->ToString();
      tape_output = NewValueNode(kNull);
      tape_output->set_abstract(nullptr);
    }
  }
  ad_param()->tape_->set_output(tape_output);
}

void IrGrad::ElimateTupleGetItem() {
  for (auto &user : ad_param()->users_.tuple_getitem_user_) {
    auto old_node = user.first;
    auto old_cnode = old_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(old_cnode);
    auto tuple_node = old_cnode->input(kIndex1);
    if (!IsPrimitiveCNode(tuple_node, prim::kPrimMakeTuple)) {
      continue;
    }
    auto index_value = GetValueNode<Int64ImmPtr>(old_cnode->input(kIndex2));
    size_t index = LongToSize(index_value->value());
    auto tuple_cnode = tuple_node->cast<CNodePtr>();
    ir_bprop_->Replace(old_node, tuple_cnode->input(index + 1), &ad_param()->users_.tuple_getitem_user_);
  }
}

void IrGrad::DoParameterReplaceByManager(bool has_sens_arg) {
  const auto &parameters = ad_param()->tape_->parameters();
  auto cell_inputs_size = cell_inputs_.size();
  auto mng = MakeManager({ad_param()->tape_}, false);
  auto tr = mng->Transact();
  for (size_t i = 0; i < cell_inputs_size; ++i) {
    (void)tr.Replace(cell_inputs_[i].first, parameters[i]);
  }
  // (Inputs, sens, weights) or (Inputs, weights)
  size_t weight_offset = cell_inputs_size;
  if (has_sens_arg) {
    weight_offset = weight_offset + 1;
  }
  for (size_t i = weight_offset; i < parameters.size(); ++i) {
    auto tensor = PyNativeAlgo::Common::GetTensorFromParam(parameters[i]);
    MS_EXCEPTION_IF_NULL(tensor);
    auto parameter = ExtractParameter(tensor);
    MS_EXCEPTION_IF_NULL(parameter);
    (void)tr.Replace(parameter, parameters[i]);
  }
  tr.Commit();
}

void IrGrad::DoParameterReplaceByUser(bool has_sens_arg, expander::bprop::UserType *user) {
  MS_EXCEPTION_IF_NULL(user);
  const auto &parameters = ad_param()->tape_->parameters();
  auto cell_inputs_size = cell_inputs_.size();
  for (size_t i = 0; i < cell_inputs_size; ++i) {
    ir_bprop_->Replace(cell_inputs_[i].first, parameters[i], user);
  }
  size_t weight_offset = cell_inputs_size;
  if (has_sens_arg) {
    weight_offset = weight_offset + 1;
  }
  for (size_t i = weight_offset; i < parameters.size(); ++i) {
    auto tensor = PyNativeAlgo::Common::GetTensorFromParam(parameters[i]);
    MS_EXCEPTION_IF_NULL(tensor);
    auto parameter = ExtractParameter(tensor);
    MS_EXCEPTION_IF_NULL(parameter);
    ir_bprop_->Replace(parameter, parameters[i], user);
  }
}

void IrGrad::ReplacePrimalParameter(bool has_sens_arg) {
  PyNativeAlgo::Common::DumpGraphIR("replace_param.ir", ad_param()->tape_);
  if (need_do_manager_replace_ || ad_param()->tape_->has_flag(kFlagIsControlFlow)) {
    MS_LOG(DEBUG) << "Do parameter replace by manager.";
    DoParameterReplaceByManager(has_sens_arg);
    need_do_manager_replace_ = false;
  } else {
    MS_LOG(DEBUG) << "Do parameter replace by user.";
    DoParameterReplaceByUser(has_sens_arg, &ad_param()->users_.dout_user_);
  }
  if (!ad_param()->reverse_users_.empty()) {
    DoParameterReplaceByUser(has_sens_arg, &ad_param()->reverse_users_);
  }
  ElimateTupleGetItem();
}

void IrGrad::UpdateTapeParameter(const tensor::BaseTensorPtr &tensor) {
  auto p = ad_param()->tape_->add_parameter();
  auto param = ExtractParameter(tensor);
  if (param == nullptr) {
    param =
      ir_bprop_->CreateTapeParameter(tensor, PyNativeAlgo::Common::SetAbstractValueToAnyValue(tensor->ToAbstract()));
  }
  MS_EXCEPTION_IF_NULL(param);
  const auto &param_info = tensor->param_info();
  if (param_info != nullptr) {
    const auto &param_name = param_info->name();
    p->set_name(param_name);
    p->debug_info()->set_name(param_name);
  }
  TraceGuard trace_guard(std::make_shared<TraceCopy>(p->debug_info()));
  p->set_default_param(tensor);
  p->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(tensor->ToAbstract()));
}
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
