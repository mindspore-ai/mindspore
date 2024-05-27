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

#include "pipeline/pynative/grad/ir/ir_bprop.h"
#include <string>
#include <vector>
#include <memory>
#include "pipeline/pynative/pynative_utils.h"
#include "include/common/utils/primitive_utils.h"
#include "pipeline/jit/ps/pass.h"
#include "ir/func_graph_cloner.h"
#include "ops/sequence_ops.h"
#include "ops/framework_ops.h"
#include "ops/structure_ops.h"
#include "ops/other_ops.h"

namespace mindspore::pynative::autograd {
namespace {
constexpr size_t kOutAndDoutNum = 2;
const mindspore::HashSet<std::string> kMonadOp = {kLoadOpName, kDependOpName, kUpdateStateOpName};
const mindspore::HashSet<std::string> kMetaFuncGraphOp{
  kPyExecuteOpName,
  kAttrMutableOpName,
  kMakeDictOpName,
};
mindspore::HashMap<std::string, FuncGraphPtr> pass_grad_graph_;

FuncGraphPtr OptimizeBpropBuilder(const FuncGraphPtr &bprop_func_graph, const GradParamPtr &grad_param) {
  PyNativeAlgo::Common::DumpGraphIR("bprop_builder_before_opt.ir", bprop_func_graph);
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_func_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_func_graph);
  auto after_opt_bg = pipeline::JitBpropGraphPass(resource, true);
  auto is_dynamic_shape_control_flow =
    grad_param->is_jit_graph && grad_param->use_dynamic_shape_process && grad_param->is_control_flow;
  if (is_dynamic_shape_control_flow) {
    for (const auto &g : manager->func_graphs()) {
      g->set_flag(kFlagJitCallGraph, true);
    }
  }
  auto abs_seq = after_opt_bg->parameters().empty()
                   ? nullptr
                   : after_opt_bg->parameters().back()->abstract()->cast<abstract::AbstractSequencePtr>();
  if (abs_seq != nullptr && !abs_seq->dynamic_len() && grad_param->is_jit_graph &&
      grad_param->use_dynamic_shape_process) {
    PyNativeAlgo::Common::ProcessTupleParam(after_opt_bg, after_opt_bg->parameters().size() - kIndex1);
  }
  PyNativeAlgo::Common::DumpGraphIR("bprop_builder_after_opt.ir", after_opt_bg);
  return after_opt_bg;
}

bool ProcessMonadNode(const PrimitivePtr &prim, const CNodePtr &cnode, const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(prim);
  if (kMonadOp.find(prim->name()) != kMonadOp.end()) {
    MS_LOG(DEBUG) << "Get monad cnode " << cnode->DebugString();
    return true;
  }
  if ((prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_MEM) || prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_IO)) &&
      (cnode->inputs().back()->abstract()->isa<abstract::AbstractMonad>())) {
    AnfNodePtrList inputs{cnode->inputs().begin(), cnode->inputs().end() - 1};
    cnode->set_inputs(inputs);
  }
  MS_EXCEPTION_IF_NULL(grad_param);
  // Jit graph contain monad op
  if (grad_param->is_jit_graph) {
    for (size_t i = 1; i < cnode->size(); ++i) {
      cnode->set_input(i, common::AnfAlgo::VisitKernelWithReturnType(cnode->input(i), 0, false,
                                                                     {prim::kPrimTupleGetItem, prim::kPrimMakeTuple})
                            .first);
    }
  }
  return false;
}

void ClearGradMetaData(const ValuePtr &value) {
  if (value->isa<tensor::BaseTensor>()) {
    auto tensor = value->cast<tensor::BaseTensorPtr>();
    tensor->set_auto_grad_meta_data(nullptr);
  } else if (value->isa<ValueSequence>()) {
    auto value_sequence = value->cast<ValueSequencePtr>();
    for (const auto &val : value_sequence->value()) {
      ClearGradMetaData(val);
    }
  }
}

// Handle bprob of op which input dtype is real number and output dtype is complex number.
// If the dtype of a gradient(din) is complex number and the input of that is real number,
// only the real part of the gradient make sense in back propagate. So we handle it by
// insert a Real() ops after the gradient.
// input: AnfNode with input of op which input dtype is real number and output dtype is complex number.
// din: CNodePtr with gradient of input.
// tape: Funcgraph witch input and din belong to.
// return: New din with inserted real op if necessarily.
AnfNodePtr HandleRealToComplex(const tensor::BaseTensorPtr &input, const AbstractBasePtr &abs, const AnfNodePtr &din,
                               const KernelGraphPtr &tape) {
  MS_EXCEPTION_IF_NULL(din);
  TypePtr din_type = din->Type();
  if (din_type == nullptr || !din_type->isa<TensorType>()) {
    return din;
  }
  din_type = din_type->cast_ptr<TensorType>()->element();
  MS_EXCEPTION_IF_NULL(din_type);
  // cppcheck-suppress unreadVariable
  if (MS_LIKELY(din_type->type_id() != kNumberTypeComplex64 && din_type->type_id() != kNumberTypeComplex128)) {
    return din;
  }

  MS_EXCEPTION_IF_NULL(input);
  TypePtr input_type = input->Dtype();
  if (input_type == nullptr) {
    return din;
  }
  if (input_type->type_id() == kNumberTypeComplex64 || input_type->type_id() == kNumberTypeComplex128) {
    return din;
  }

  AnfNodePtr new_din = tape->FuncGraph::NewCNode({NewValueNode(prim::kPrimReal), din});
  AbstractBasePtr real_abs =
    std::make_shared<abstract::AbstractTensor>(abstract::AbstractTensor(input_type, abs->GetShapeTrack()));
  new_din->set_abstract(real_abs);
  return new_din;
}

void PlantFuncGradBpropGraphDout(const GradParamPtr &grad_param, const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(grad_param);
  if (!grad_param->is_func_grad) {
    return;
  }
  // Plant dout tuple or dict
  if (graph->parameters().back()->abstract()->isa<abstract::AbstractSequence>()) {
    PyNativeAlgo::Common::ProcessTupleParam(graph, grad_param->input_size);
  } else if (graph->parameters().back()->abstract()->isa<abstract::AbstractDictionary>()) {
    PyNativeAlgo::Common::ProcessDictParam(graph, grad_param->input_size);
  }
}
}  // namespace

void ClearAutoGradCache() {
  pass_grad_graph_.clear();
  bprop_pass::ClearCache();
  PyNativeAlgo::AutoGrad::ClearAutoGradStaticCache();
}

std::pair<bool, FuncGraphPtr> IrBprop::GetBpropGraph(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  const auto it = pass_grad_graph_.find(grad_param->graph_cache_key);
  bool cache_hit = (it != pass_grad_graph_.end());
  if (grad_param->is_control_flow || grad_param->is_jit_self_dynamic_shape) {
    MS_LOG(DEBUG) << "Get control flow graph or dynamic shape";
    return std::make_pair(cache_hit, GetBpropGraphFromFprop(grad_param));
  }
  return std::make_pair(cache_hit, GetBpropGraphFromExpander(grad_param));
}

void IrBprop::BuildCustomBpropCNode(const CNodePtr &cnode, const PrimitivePtr &prim, std::vector<CNodePtr> *outputs) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(DEBUG) << "Try build custom bprop: " << prim->name();
  {
    py::gil_scoped_acquire gil;
    auto prim_py = prim->cast<PrimitivePyPtr>();
    if (prim_py == nullptr) {
      MS_LOG(DEBUG) << "Prim is not PrimitivePy, can not find python bprop";
      return;
    }
    py::function fn = prim_py->GetBpropFunction();
    if (py::isinstance<py::none>(fn)) {
      fn = GetBpropFunction(prim->name());
    }
    if (!fn || py::isinstance<py::none>(fn)) {
      MS_LOG(INFO) << "Can not find bprop function for " << prim->name() << ". fn: " << ConvertPyObjToString(fn);
      return;
    }
    (void)prim_py->AddBackwardHookFn(0, fn);
    (void)prim_py->AddAttr("custom_op_bprop", MakeValue(true));
  }
  BuildBPropCutCNode(cnode, prim, outputs);
}

void IrBprop::BuildBPropCutCNode(const CNodePtr &cnode, const PrimitivePtr &prim, std::vector<CNodePtr> *outputs,
                                 bool is_need_recompute) {
  MS_EXCEPTION_IF_NULL(prim);
  auto bprop_cut = PyNativeAlgo::AutoGrad::BuildBpropCutPrim(prim, is_need_recompute);

  // Create gradient outputs cnode
  AnfNodePtrList inputs{NewValueNode(bprop_cut)};
  for (size_t i = 1; i < cnode->size() - kOutAndDoutNum; ++i) {
    (void)inputs.emplace_back(cnode->input(i));
  }
  if (!is_need_recompute) {
    // If not recompute, we should add out as bprop input.
    (void)inputs.emplace_back(cnode->input(cnode->size() - kOutAndDoutNum));
  }
  (void)inputs.emplace_back(cnode->input(cnode->size() - 1));

  auto bprop_cut_cnode = ad_param_->tape_->FuncGraph::NewCNode(inputs);
  AbstractBasePtrList abs_list;
  // Only add last input dout to user.
  AddUser(cnode->input(cnode->size() - 1), bprop_cut_cnode, bprop_cut_cnode->size() - 1);
  for (size_t i = 1; i < cnode->size() - kOutAndDoutNum; ++i) {
    // Input may be parameter, we need add to user map.
    AddUser(cnode->input(i), bprop_cut_cnode, i);
    auto din = ad_param_->tape_->FuncGraph::NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), bprop_cut_cnode, NewValueNode(static_cast<int64_t>(i - 1))});
    MS_EXCEPTION_IF_NULL(cnode->input(i)->abstract());
    din->set_abstract(cnode->input(i)->abstract());
    (void)abs_list.emplace_back(cnode->input(i)->abstract());
    (void)outputs->emplace_back(din);
  }
  bprop_cut_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
  ad_param_->tape_->set_flag(kFlagPyNativeBpropGraphWithBpropCut, true);
  bprop_graph_run_by_single_op_ = true;
}

AnfNodePtr IrBprop::MapParameter(const ValuePtr &value, const abstract::AbstractBasePtr &abs) {
  if (value->isa<tensor::BaseTensor>()) {
    const auto &tensor = value->cast<tensor::BaseTensorPtr>();
    const auto &auto_grad_meta_data = tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    const auto &param = auto_grad_meta_data->parameter();
    if (param != nullptr) {
      // In dynamic shape scenario, abs my be need change
      param->set_abstract(abs);
      return param;
    }
    set_bprop_graph_run_by_single_op(auto_grad_meta_data->is_register_hook());
    if (auto_grad_meta_data->input_type() == InputType::kParameter &&
        PyNativeAlgo::Common::IsParamRequiresGrad(tensor)) {
      return AddParameterNode(tensor, abs);
    }
    return PyNativeAlgo::Common::CreateValueNodeByValue(value, abs);
  } else if (value->isa<ValueSequence>()) {
    const auto &val_seq = value->cast<ValueSequencePtr>()->value();
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (val_seq.size() != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Get value sequence size " << val_seq.size() << " not equal to abstract size "
                        << abs_seq->size();
    }
    AnfNodePtrList inputs;
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t i = 0; i < val_seq.size(); ++i) {
      (void)inputs.emplace_back(MapParameter(val_seq[i], abs_seq->elements()[i]));
    }
    auto cnode = ad_param_->tape_->FuncGraph::NewCNode(inputs);
    // For replacing fg parameter by user
    for (size_t i = 1; i < inputs.size(); ++i) {
      AddUser(inputs[i], cnode, i);
    }
    cnode->set_abstract(abs);
    return cnode;
  } else if (value->isa<tensor::COOTensor>()) {
    const auto &coo_tensor = value->cast<tensor::COOTensorPtr>();
    return MapParameter(coo_tensor->GetIndices(), abs);
  } else if (value->isa<tensor::CSRTensor>()) {
    const auto &csr_tensor = value->cast<tensor::CSRTensorPtr>();
    return MapParameter(csr_tensor->GetIndices(), abs);
  } else {
    return PyNativeAlgo::Common::CreateValueNodeByValue(value, abs);
  }
}

ParameterPtr IrBprop::AddParameterNode(const tensor::BaseTensorPtr &tensor, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto param = CreateTapeParameter(tensor, abs);
  auto zeros_like_dout = PyNativeAlgo::AutoGrad::BuildSpecialNode(
    ad_param_->tape_, PyNativeAlgo::AutoGrad::GetFakeZeroTensor(), param->abstract(), SpecialType::kZerosLikeType);
  auto func_node = std::make_shared<IrFunctionNode>(ad_param_->tape_, zeros_like_dout);
  auto input_adjoint = std::make_shared<IrVariable>(func_node, tensor, true);
  (void)ad_param_->variable_adjoint_set_.insert(input_adjoint);
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
  auto_grad_meta_data->set_variable(input_adjoint);
  (void)ad_param_->weights_used_in_graph_.emplace_back(param);
  return param;
}

ParameterPtr IrBprop::CreateTapeParameter(const tensor::BaseTensorPtr &tensor, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(abs);
  auto param = ad_param_->fg_->add_parameter();
  param->set_abstract(abs);
  if (tensor->is_parameter()) {
    param->set_default_param(tensor);
  }
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  if (auto_grad_meta_data == nullptr) {
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    tensor->set_auto_grad_meta_data(auto_grad_meta_data);
  }
  auto_grad_meta_data->set_input_type(InputType::kParameter);
  auto_grad_meta_data->set_parameter(param);
  return param;
}

void IrBprop::UpdateNextEdges(const VariablePtr &variable, const std::vector<CNodePtr> &dins,
                              const ValuePtrList &inputs_value, const abstract::AbstractBasePtrList &abs,
                              const string &op_name) {
  size_t input_size = inputs_value.size();
  if (dins.size() != input_size) {
    MS_LOG(EXCEPTION) << "The size of dins " << dins.size() << " is not same as input_value " << input_size;
  }
  const auto &fn = variable->ir_function_node();
  for (size_t i = 0; i < input_size; ++i) {
    auto din = dins[i];
    MS_EXCEPTION_IF_NULL(din);
    MS_LOG(DEBUG) << "Input arg id: " << PyNativeAlgo::Common::GetIdByValue(inputs_value[i]) << ", din "
                  << din->DebugString();
#ifndef ENABLE_TEST
    // VM no need run pass
    din = pass_forward_->PassForDin(din, op_name, false);
#endif
    UpdateNextEdge(fn, din, inputs_value[i], abs[i]);
  }
  if (fn->next_edges().empty()) {
    variable->set_is_need_grad(false);
  }
  MS_LOG(DEBUG) << "Finish update next edges for variable: " << variable->ToString();
}

void IrBprop::AddUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  MS_EXCEPTION_IF_NULL(ad_param_);
  (void)ad_param_->users_.dout_user_[node].emplace_back(user, index);
}

void IrBprop::AddReverseUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  (void)ad_param_->reverse_users_[node].emplace_back(user, index);
}

void IrBprop::BackPropagate() {
  UpdateLazyUser();
  const auto &last_node_reverse_iter = GetLastNodeReverseIter();
#ifndef ENABLE_TEST
  SeenNum seen = NewSeenGeneration();
#endif
  MS_LOG(DEBUG) << "Is running recompute grad " << is_run_recompute_;
  for (auto iter = last_node_reverse_iter; iter != ad_param_->variable_adjoint_set_.rend(); ++iter) {
    const auto &variable = *iter;
    if (!variable->is_need_propagate() || !variable->is_need_grad()) {
      MS_LOG(DEBUG) << "No need grad, variable is: " << variable->ToString();
      continue;
    }
    if (static_cast<bool>(MS_UNLIKELY(variable->is_fake_bprop()))) {
      MS_LOG(EXCEPTION) << "Illegal primitive " << variable->fake_prim_name() << "'s bprop not defined";
    }
    MS_LOG(DEBUG) << "Begin backpropagate: " << variable->ToString();
    const auto &fn = variable->ir_function_node();
    // If zeroslike not used in funcgraph, we need replace the zeroslike placeholder with real zeroslike value.
    if (static_cast<bool>(MS_UNLIKELY(PyNativeAlgo::AutoGrad::IsZerosLikeNode(fn->accumulate_dout())))) {
      fn->set_accumulate_dout(PyNativeAlgo::AutoGrad::BuildSpecialNode(
        fn->tape(), variable->out_value(), fn->accumulate_dout()->abstract(), SpecialType::kZerosLikeType));
    }
    // If register hook by weight, and weight in recompute cell.So, hook will execute, which is not expect.
    if (!is_run_recompute_) {
      fn->set_accumulate_dout(pass_forward_->PassBackwardHook(variable->out_value(), fn->accumulate_dout()));
    }
    // Replace real dout to fake dout, update replace result to eliminate tuplegetitem
    // when accumulate_dout is tuplegetitem
    Replace(fn->fake_dout(), fn->accumulate_dout(), &ad_param_->users_.dout_user_, true);
    // replace edges which exist fake dout
    fn->ReplaceEdges();
    const auto &next_edges = fn->next_edges();
    for (const auto &next_edge : next_edges) {
      const auto &last_variable = next_edge.first;
      const auto &din = next_edge.second;
#ifndef ENABLE_TEST
      // VM no need run pass
      pass_forward_->ConvertMakeTupleInputToDynamicInput(din, seen, bprop_graph_run_by_single_op_);
#endif
      last_variable->ir_function_node()->UpdateAccumulativeDout(din);
      last_variable->set_is_need_propagate(true);
    }
  }
  MS_LOG(DEBUG) << "End BackPropagate";
}

OrderedSet<IrVariablePtr>::reverse_iterator IrBprop::GetLastNodeReverseIter() {
  for (auto iter = ad_param_->variable_adjoint_set_.rbegin(); iter != ad_param_->variable_adjoint_set_.rend(); ++iter) {
    if (*iter == ad_param_->last_variable_) {
      ad_param_->last_variable_->set_is_need_propagate(true);
      return iter;
    }
  }
  return ad_param_->variable_adjoint_set_.rend();
}

AbstractBasePtr IrBprop::BuildForwardLastNode() {
  MS_LOG(DEBUG) << "Process last node info " << PyNativeAlgo::Common::GetIdByValue(ad_param_->sens_value_);
  auto zeros_like_node = PyNativeAlgo::AutoGrad::BuildSpecialNode(ad_param_->tape_, ad_param_->sens_value_, nullptr,
                                                                  SpecialType::kZerosLikeType);
  auto fn = std::make_shared<IrFunctionNode>(ad_param_->tape_, zeros_like_node);
  auto sens_variable = std::make_shared<IrVariable>(fn, ad_param_->sens_value_);
  if (ad_param_->sens_value_->isa<tensor::BaseTensor>()) {
    const auto &sens_tensor = ad_param_->sens_value_->cast<tensor::BaseTensorPtr>();
    const auto &auto_grad_meta_data = sens_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    if (PyNativeAlgo::Common::IsConstant(auto_grad_meta_data->input_type())) {
      sens_variable->set_is_need_grad(false);
    }
  }
  UpdateNextEdge(fn, zeros_like_node, ad_param_->sens_value_, fn->accumulate_dout()->abstract());
  (void)ad_param_->variable_adjoint_set_.insert(sens_variable);
  ad_param_->last_variable_ = sens_variable;
  return fn->accumulate_dout()->abstract();
}

FuncGraphPtr IrBprop::GetBpropGraphFromFprop(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  FuncGraphPtr after_opt_fg = nullptr;
  // Find ad graph in cache
  const auto it = pass_grad_graph_.find(grad_param->graph_cache_key);
  bool cache_hit = (it != pass_grad_graph_.end());
  if (cache_hit) {
    MS_LOG(DEBUG) << "Get ad grad graph by cache";
    after_opt_fg = BasicClone(it->second);
  } else {
    auto bprop_builder = std::make_shared<FuncGraph>();
    bprop_builder->debug_info()->set_name("bprop_builder");

    AnfNodePtrList fprop_app_inputs{NewValueNode(grad_param->fg)};
    for (const auto &abs : grad_param->op_grad_info->input_abs) {
      auto param = bprop_builder->add_parameter();
      param->set_abstract(abs);
      (void)fprop_app_inputs.emplace_back(param);
    }
    auto fprop_app = bprop_builder->NewCNode(fprop_app_inputs);
    // Get bprop from fprop_fg, it is 2th output of fprop_fg
    auto get_bprop = bprop_builder->NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), fprop_app, NewValueNode(static_cast<int64_t>(kIndex1))});

    AnfNodePtrList node_list{get_bprop};
    auto dout = bprop_builder->add_parameter();
    dout->set_abstract(grad_param->op_grad_info->out_abs);
    (void)node_list.emplace_back(dout);
    auto call_bprop = bprop_builder->NewCNode(node_list);

    AnfNodePtrList actual_out{NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < grad_param->input_size; ++i) {
      // Index 0 env, skip
      auto out =
        bprop_builder->NewCNode({NewValueNode(prim::kPrimTupleGetItem), call_bprop, NewValueNode(SizeToLong(i + 1))});
      (void)actual_out.emplace_back(out);
    }
    bprop_builder->set_output(bprop_builder->NewCNode(actual_out));
    // Call pass for optimize graph, such as inline
    after_opt_fg = OptimizeBpropBuilder(bprop_builder, grad_param);
    PlantFuncGradBpropGraphDout(grad_param, after_opt_fg);
    if (grad_param->is_func_grad && grad_param->is_control_flow) {
      after_opt_fg = LiftingClone(after_opt_fg);
    }
    if (grad_param->is_jit_graph || !grad_param->use_dynamic_shape_process) {
      pass_grad_graph_[grad_param->graph_cache_key] = BasicClone(after_opt_fg);
    }
  }
  return after_opt_fg;
}

FuncGraphPtr IrBprop::GetBpropGraphFromExpander(const GradParamPtr &grad_param) {
  // Find ad graph in cache
  if (grad_param->is_jit_graph || !grad_param->use_dynamic_shape_process) {
    const auto it = pass_grad_graph_.find(grad_param->graph_cache_key);
    if (it != pass_grad_graph_.end()) {
      MS_LOG(DEBUG) << "Get ad grad graph by cache";
      return BasicClone(it->second);
    }
  } else {
    pass_grad_graph_.clear();
  }

  // Create new ad param for graph ad
  PyNativeAlgo::Common::DumpGraphIR("ad_input_graph.ir", grad_param->fg);
  auto current_ad_param = ad_param_;
  ad_param_ = std::make_shared<AdParam>();
  ad_param_->tape_->debug_info()->set_name("ad_graph");
  bprop_graph_run_by_single_op_ = bprop_graph_run_by_single_op_ || grad_param->use_dynamic_shape_process;

  GradGraphByExpander(grad_param);

  if (ad_param_->last_node_ != nullptr) {
    // Set dout parameter
    const auto last_prim = GetCNodePrimitive(ad_param_->last_node_);
    if (kMonadOp.find(last_prim->name()) != kMonadOp.end()) {
      ad_param_->last_node_ = common::AnfAlgo::VisitKernelWithReturnType(
                                ad_param_->last_node_, 0, false, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple})
                                .first;
    }
    if (ad_param_->anfnode_to_variable_adjoint_.count(ad_param_->last_node_) == 0) {
      MS_LOG(EXCEPTION) << "Can not find last node" << ad_param_->last_node_->DebugString();
    }
    ad_param_->last_variable_ = ad_param_->anfnode_to_variable_adjoint_[ad_param_->last_node_];
    auto ad_graph_dout = ad_param_->tape_->add_parameter();
    ad_graph_dout->set_abstract(ad_param_->last_node_->abstract());
    ad_param_->last_variable_->ir_function_node()->UpdateAccumulativeDout(ad_graph_dout);
    (void)BackPropagate();
  } else {
    // Just have a return node
    auto ad_graph_dout = ad_param_->tape_->add_parameter();
    ad_graph_dout->set_abstract(grad_param->fg->output()->abstract());
    ad_graph_dout->debug_info()->set_name("sens");
    ad_param_->sens_value_ = grad_param->op_grad_info->out_value;
    (void)BuildForwardLastNode();
    // Update dout
    MS_EXCEPTION_IF_NULL(ad_param_->last_variable_);
    if (ad_param_->last_variable_->is_need_grad()) {
      ad_param_->last_variable_->ir_function_node()->UpdateAccumulativeDout(ad_graph_dout);
    }
    (void)BackPropagate();
  }

  AnfNodePtrList outputs{NewValueNode(prim::kPrimMakeTuple)};
  abstract::AbstractBasePtrList out_abs_list;
  for (const auto &node : grad_param->fg->parameters()) {
    (void)outputs.emplace_back(ad_param_->anfnode_to_variable_adjoint_.at(node)->RealDout());
    (void)out_abs_list.emplace_back(outputs.back()->abstract());
  }
  auto ad_graph_out = ad_param_->tape_->FuncGraph::NewCNode(outputs);
  ad_graph_out->set_abstract(std::make_shared<abstract::AbstractTuple>(out_abs_list));
  ad_param_->tape_->set_output(ad_graph_out);
  auto ad_graph = ad_param_->tape_;
  auto abs_seq = ad_graph->parameters().empty()
                   ? nullptr
                   : ad_graph->parameters().back()->abstract()->cast<abstract::AbstractSequencePtr>();
  if (abs_seq != nullptr && !abs_seq->dynamic_len() && grad_param->is_jit_graph &&
      grad_param->use_dynamic_shape_process) {
    auto manager = MakeManager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(ad_graph);
    PyNativeAlgo::Common::ProcessTupleParam(ad_graph, ad_graph->parameters().size() - kIndex1);
  }
  PyNativeAlgo::Common::DumpGraphIR("ad_output_graph.ir", ad_graph);

  // Plant dout tuple
  PlantFuncGradBpropGraphDout(grad_param, ad_graph);

  // Save ad graph in cache
  if (grad_param->is_jit_graph || !grad_param->use_dynamic_shape_process) {
    pass_grad_graph_[grad_param->graph_cache_key] = BasicClone(ad_graph);
  }
  // Replace cnode with valuenode for reduce compute
  bool jit_by_value = grad_param->is_jit_graph && grad_by_value_;
  if (jit_by_value) {
    PyNativeAlgo::Common::ReplaceCNodeWithValueNode(ad_graph);
  }
  // Restore ad param
  ad_param_ = current_ad_param;
  return ad_graph;
}

void IrBprop::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node, expander::bprop::UserType *user,
                      bool need_update) {
  MS_EXCEPTION_IF_NULL(user);
  if (user->find(old_node) == user->end()) {
    return;
  }
  const auto &old_node_users = (*user)[old_node];
  for (const auto &pair_node : old_node_users) {
    auto cnode = pair_node.first.lock();
    if (cnode == nullptr) {
      continue;
    }
    size_t index = pair_node.second;
    if (index >= cnode->size()) {
      // After convert attr cnode input will less
      if (auto v = cnode->GetAttr(kAttrConvertAttrNode); v != nullptr) {
        index -= GetValue<size_t>(v);
      } else {
        MS_LOG(EXCEPTION) << "exception for index: " << index << "greater than cnode size: " << cnode->size();
      }
    }
    cnode->set_input(index, new_node);
    if (need_update && IsPrimitiveCNode(new_node, prim::kPrimTupleGetItem)) {
      AddTupleGetItemUser(new_node, cnode, index);
    }
  }
}

void IrBprop::GradGraphByExpander(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  if (pass_forward_->need_reverse_graph()) {
    pass_forward_->ReversePassFuncGraph(grad_param->fg);
  }

  // First handle parameters
  CreateParameterAdjoint(grad_param);

  // Second handle cnodes
  const auto &order = TopoSort(grad_param->fg->output());
  for (const auto &node : order) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitive(cnode);
    if (prim == nullptr) {
      MS_LOG(EXCEPTION) << "Should be primitive, but: " << cnode->DebugString();
    }
    ad_param_->last_node_ = cnode;
    if (ProcessMonadNode(prim, cnode, grad_param) || IsPrimitiveEquals(prim, prim::kPrimStopGradient)) {
      continue;
    }
    MS_LOG(DEBUG) << "Get cnode " << cnode->DebugString() << ", " << cnode->fullname_with_scope();
    ValuePtrList inputs_value;
    AnfNodePtrList cnode_inputs;
    PrepareGradCNodeInputs(prim, cnode, &inputs_value, &cnode_inputs);
    // Do grad for every cnode
    GradCNode(prim, cnode, grad_param, inputs_value, &cnode_inputs);
  }
}

void IrBprop::CreateParameterAdjoint(const GradParamPtr &grad_param) const {
  auto &graph_parameters = grad_param->fg->parameters();
  if (graph_parameters.size() != grad_param->input_size) {
    MS_LOG(EXCEPTION) << "Parameters size " << graph_parameters.size() << " is not equal to graph input size "
                      << grad_param->input_size;
  }
  for (size_t i = 0; i < graph_parameters.size(); ++i) {
    MS_LOG(DEBUG) << "Get param " << graph_parameters[i]->DebugString();
    ParameterPtr param = ad_param_->tape_->add_parameter();
    param->set_abstract(graph_parameters[i]->abstract());
    auto zeros_like_dout =
      PyNativeAlgo::AutoGrad::BuildSpecialNode(ad_param_->tape_, PyNativeAlgo::AutoGrad::GetFakeZeroTensor(),
                                               graph_parameters[i]->abstract(), SpecialType::kZerosLikeType);
    auto func_node = std::make_shared<IrFunctionNode>(ad_param_->tape_, zeros_like_dout);
    // Copy to avoid corrupt real input grad info.
    auto op_arg = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(grad_param->op_grad_info->input_value[i]);
    ClearGradMetaData(op_arg);
    auto adjoint = std::make_shared<IrVariable>(func_node, op_arg, true);
    adjoint->set_k_node(param);
    PyNativeAlgo::AutoGrad::SetGradMetaData(op_arg, adjoint, graph_parameters[i]->cast<ParameterPtr>());
    (void)ad_param_->variable_adjoint_set_.insert(adjoint);
    (void)ad_param_->anfnode_to_variable_adjoint_.insert(std::make_pair(graph_parameters[i], adjoint));
  }
}

void IrBprop::PrepareGradCNodeInputs(const PrimitivePtr &prim, const CNodePtr &cnode, ValuePtrList *inputs_value,
                                     AnfNodePtrList *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(inputs_value);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  (void)cnode_inputs->emplace_back(std::make_shared<ValueNode>(prim));
  *inputs_value = GetInputArgs(cnode, cnode_inputs);
  pass_forward_->ReversePassCNode(cnode, inputs_value, cnode_inputs);
}

ValuePtrList IrBprop::GetInputArgs(const CNodePtr &cnode, AnfNodePtrList *cnode_inputs) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  ValuePtrList input_value;
  for (size_t i = 1; i < cnode->size(); ++i) {
    const auto &input_node = cnode->input(i);
    // Find knode and out value
    const auto it = ad_param_->anfnode_to_variable_adjoint_.find(input_node);
    if (it != ad_param_->anfnode_to_variable_adjoint_.end()) {
      (void)cnode_inputs->emplace_back(it->second->k_node());
      (void)input_value.emplace_back(it->second->out_value());
      continue;
    }
    if (input_node->isa<ValueNode>()) {
      auto v_node = input_node->cast<ValueNodePtr>();
      auto v = v_node->value();
      if (v != nullptr && v->isa<tensor::BaseTensor>()) {
        const auto &t = v->cast<tensor::BaseTensorPtr>();
        const auto &grad_meta = t->auto_grad_meta_data();
        // Jit forward graph has no parameters(input is tuple or constant), so input used in graph as valuenode, but it
        // is used by tape_ as parameter also
        if (grad_meta != nullptr && PyNativeAlgo::Common::IsParam(grad_meta->input_type())) {
          auto new_tensor = std::make_shared<tensor::Tensor>(t->data_type(), t->shape(), t->data_ptr());
          new_tensor->set_device_address(t->device_address());
          v = new_tensor;
        }
      }
      (void)PyNativeAlgo::Common::SetValueGradInfo(v, nullptr, InputType::kConstant);
      // In case of jit forward graph and pynative bprop graph used same valuenode
      auto new_v_node = PyNativeAlgo::Common::CreateValueNodeByValue(v, v_node->abstract());
      (void)cnode_inputs->emplace_back(new_v_node);
      (void)input_value.emplace_back(v);
    } else {
      // Make Fake value
      auto v = MakeValue<int64_t>(0);
      (void)cnode_inputs->emplace_back(PyNativeAlgo::Common::CreateValueNodeByValue(v, input_node->abstract()));
      (void)input_value.emplace_back(v);
      MS_LOG(DEBUG) << "Get input node " << input_node->DebugString();
    }
  }
  return input_value;
}

void IrBprop::GradCNode(const PrimitivePtr &prim, const CNodePtr &cnode, const GradParamPtr &grad_param,
                        const ValuePtrList &inputs_value, AnfNodePtrList *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(cnode);
  bool jit_by_value = grad_param->is_jit_graph && grad_by_value_;
  if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple) || IsPrimitiveEquals(prim, prim::kPrimMakeList)) {
    (void)BuildKNodeForMakeTuple(cnode);
    return;
  }
  if (IsPrimitiveEquals(prim, prim::kPrimTupleGetItem)) {
    (void)BuildKNodeForTupleGetItem(cnode);
    return;
  }
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  auto k_node = GetKnode(prim, cnode, *cnode_inputs, jit_by_value);
  if (bprop_graph_run_by_single_op_ && !IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) &&
      std::any_of(cnode->inputs().begin() + 1, cnode->inputs().end(), [](const AnfNodePtr &node) {
        MS_EXCEPTION_IF_NULL(node->abstract());
        return node->abstract()->isa<abstract::AbstractSequence>();
      })) {
    k_node->cast<CNodePtr>()->AddAttr(kAttrIsPyboostTupleInput, MakeValue(true));
  }
  MS_LOG(DEBUG) << "Build knode " << k_node->DebugString();
  // Set out
  auto out = PyNativeAlgo::Common::CreatOutputTensorValueByAbstract(cnode->abstract());
  (void)cnode_inputs->emplace_back(k_node);
  // Set dout
  AnfNodePtr dout = PyNativeAlgo::AutoGrad::BuildSpecialNode(
    ad_param_->tape_, PyNativeAlgo::AutoGrad::GetFakeZeroTensor(), cnode->abstract(), SpecialType::kZerosLikeType);
  (void)cnode_inputs->emplace_back(dout);
  auto input_node = ad_param_->tape_->FuncGraph::NewCNode(*cnode_inputs);
  input_node->set_abstract(cnode->abstract());

  std::vector<CNodePtr> outputs;
  // Get bprop by expander
  auto ret = BpropExpander(&outputs, &ad_param_->users_).Run(input_node);
  if (!ret || outputs.empty()) {
    // Get bprop by python custom
    MS_LOG(DEBUG) << "Expander has no bprop of this node: " << input_node->DebugString();
    BuildCustomBpropCNode(input_node, prim, &outputs);
  }

  auto fn = std::make_shared<IrFunctionNode>(ad_param_->tape_, dout);
  auto variable_adjoint = std::make_shared<IrVariable>(fn, out);
  variable_adjoint->set_k_node(k_node);
  // Get bprop by fake bprop
  if (outputs.empty()) {
    MS_LOG(DEBUG) << "Build fake bprop for this node: " << input_node->DebugString();
    PyNativeAlgo::AutoGrad::BuildFakeBpropCNode(input_node, &outputs);
    variable_adjoint->set_is_fake_bprop(true);
    variable_adjoint->set_fake_prim_name(prim->name());
  }
  // Create current op node din edge
  AbstractBasePtrList input_abs;
  for (size_t i = 1; i < cnode->size(); ++i) {
    (void)input_abs.emplace_back(cnode->input(i)->abstract());
  }
  UpdateNextEdges(variable_adjoint, outputs, inputs_value, input_abs);
  PyNativeAlgo::AutoGrad::SetGradMetaData(out, variable_adjoint);
  (void)ad_param_->anfnode_to_variable_adjoint_.insert(std::make_pair(cnode, variable_adjoint));
  (void)ad_param_->variable_adjoint_set_.insert(variable_adjoint);
}

AnfNodePtr IrBprop::BuildKNodeForMakeTuple(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_LOG(DEBUG) << "Build knode for MakeTuple " << input_node->DebugString();
  const auto &cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfNodePtrList inputs{NewValueNode(prim::kPrimMakeTuple)};
  ValuePtrList input_value;
  AbstractBasePtrList input_abs;
  for (size_t i = 1; i < cnode->size(); ++i) {
    (void)inputs.emplace_back(BuildKNodeForCNodeInput(cnode->input(i)));
    if (cnode->input(i)->isa<CNode>() || cnode->input(i)->isa<Parameter>()) {
      const auto input_adjoint_iter = ad_param_->anfnode_to_variable_adjoint_.find(cnode->input(i));
      if (input_adjoint_iter == ad_param_->anfnode_to_variable_adjoint_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find input in adjoint map, inp: " << cnode->input(i)->DebugString();
      }
      (void)input_value.emplace_back(input_adjoint_iter->second->out_value());
      (void)input_abs.emplace_back(cnode->input(i)->abstract());
    } else {
      auto value_node = cnode->input(i)->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      (void)input_value.emplace_back(value_node->value());
      (void)input_abs.emplace_back(value_node->abstract());
    }
  }
  auto out_value = MakeValue(input_value);
  AnfNodePtr dout = PyNativeAlgo::AutoGrad::BuildSpecialNode(ad_param_->tape_, out_value, input_node->abstract(),
                                                             SpecialType::kZerosLikeType);
  auto fn = std::make_shared<IrFunctionNode>(ad_param_->tape_, dout);
  auto variable_adjoint = std::make_shared<IrVariable>(fn, out_value);
  auto k_node = ad_param_->tape_->FuncGraph::NewCNode(inputs);
  k_node->set_abstract(input_node->abstract());
  variable_adjoint->set_k_node(k_node);
  // Create dout for maketuple
  std::vector<CNodePtr> make_tuple_dout;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto d = ad_param_->tape_->FuncGraph::NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), dout, NewValueNode(SizeToLong(i - 1))});
    d->set_abstract(cnode->input(i)->abstract());
    (void)make_tuple_dout.emplace_back(d);
    AddUser(dout, d, 1);
  }
  UpdateNextEdges(variable_adjoint, make_tuple_dout, input_value, input_abs);
  (void)ad_param_->anfnode_to_variable_adjoint_.insert(std::make_pair(input_node, variable_adjoint));
  (void)ad_param_->variable_adjoint_set_.insert(variable_adjoint);
  return k_node;
}

AnfNodePtr IrBprop::BuildKNodeForCNodeInput(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_node->isa<CNode>()) {
    const auto input_adjoint_iter = ad_param_->anfnode_to_variable_adjoint_.find(input_node);
    if (input_adjoint_iter == ad_param_->anfnode_to_variable_adjoint_.end()) {
      if (IsPrimitiveCNode(input_node, prim::kPrimMakeTuple)) {
        return BuildKNodeForMakeTuple(input_node);
      } else if (IsPrimitiveCNode(input_node, prim::kPrimTupleGetItem)) {
        return BuildKNodeForTupleGetItem(input_node);
      }
      MS_LOG(EXCEPTION) << "Can not find input in adjoint map, inp: " << input_node->DebugString();
    }
    return input_adjoint_iter->second->k_node();
  } else {
    // Tuple sens will come in
    if (input_node->isa<Parameter>()) {
      const auto input_adjoint_iter = ad_param_->anfnode_to_variable_adjoint_.find(input_node);
      if (input_adjoint_iter != ad_param_->anfnode_to_variable_adjoint_.end() &&
          input_adjoint_iter->second->k_node() != nullptr) {
        return input_adjoint_iter->second->k_node();
      }
    }
    return input_node;
  }
}

AnfNodePtr IrBprop::BuildKNodeForTupleGetItem(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_LOG(DEBUG) << "Build knode for TupleGetItem " << input_node->DebugString();
  const auto &tuple_item_cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_item_cnode);
  // Find make tuple or sens(tuple) node for get out value
  const auto input_adjoint_iter = ad_param_->anfnode_to_variable_adjoint_.find(tuple_item_cnode->input(kIndex1));
  if (input_adjoint_iter == ad_param_->anfnode_to_variable_adjoint_.end()) {
    MS_LOG(EXCEPTION) << "Cannot find input in adjoint map, inp: " << tuple_item_cnode->input(kIndex1)->DebugString();
  }
  const auto &v_tuple = input_adjoint_iter->second->out_value()->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(v_tuple);
  auto index_value = GetValueNode<Int64ImmPtr>(tuple_item_cnode->input(kIndex2));
  auto index_value_int = LongToSize(index_value->value());
  auto out_value = (*v_tuple)[index_value_int];
  MS_EXCEPTION_IF_NULL(out_value);
  AnfNodePtr dout = PyNativeAlgo::AutoGrad::BuildSpecialNode(ad_param_->tape_, out_value, input_node->abstract(),
                                                             SpecialType::kZerosLikeType);
  auto fn = std::make_shared<IrFunctionNode>(ad_param_->tape_, dout);
  auto variable_adjoint = std::make_shared<IrVariable>(fn, out_value);

  AnfNodePtrList inputs{NewValueNode(prim::kPrimTupleGetItem)};
  // Get make tuple knode
  (void)inputs.emplace_back(BuildKNodeForCNodeInput(tuple_item_cnode->input(kIndex1)));
  // Get index knode
  (void)inputs.emplace_back(BuildKNodeForCNodeInput(tuple_item_cnode->input(kIndex2)));
  auto k_node = ad_param_->tape_->FuncGraph::NewCNode(inputs);
  k_node->set_abstract(input_node->abstract());
  variable_adjoint->set_k_node(k_node);
  // Create dout for tuplegetitem
  AnfNodePtrList tuple_getitem_dout{NewValueNode(prim::kPrimMakeTuple)};
  const auto &abs_tuple = tuple_item_cnode->input(kIndex1)->abstract()->cast<abstract::AbstractSequencePtr>();
  for (size_t i = 0; i < v_tuple->size(); ++i) {
    const auto &v = v_tuple->value()[i];
    if (i == index_value_int) {
      (void)tuple_getitem_dout.emplace_back(dout);
    } else {
      (void)tuple_getitem_dout.emplace_back(PyNativeAlgo::AutoGrad::BuildSpecialNode(
        ad_param_->tape_, v, abs_tuple->elements()[i], SpecialType::kZerosLikeType));
    }
  }
  CNodePtr tuple_getitem_dout_value = ad_param_->tape_->FuncGraph::NewCNode(tuple_getitem_dout);
  tuple_getitem_dout_value->set_abstract(tuple_item_cnode->input(kIndex1)->abstract());
  auto index_dout_value =
    PyNativeAlgo::AutoGrad::BuildSpecialNode(ad_param_->tape_, index_value,
                                             tuple_item_cnode->input(kIndex1)->abstract(), SpecialType::kZerosLikeType)
      ->cast<CNodePtr>();
  UpdateNextEdges(variable_adjoint, {tuple_getitem_dout_value, index_dout_value}, {v_tuple, index_value},
                  {tuple_item_cnode->input(kIndex1)->abstract(), tuple_item_cnode->input(kIndex2)->abstract()});
  AddUser(dout, tuple_getitem_dout_value, index_value_int + 1);
  (void)ad_param_->anfnode_to_variable_adjoint_.insert(std::make_pair(input_node, variable_adjoint));
  (void)ad_param_->variable_adjoint_set_.insert(variable_adjoint);
  return k_node;
}

AnfNodePtr IrBprop::GetKnode(const PrimitivePtr &prim, const CNodePtr &cnode, const AnfNodePtrList &cnode_inputs,
                             bool jit_by_value) {
  if (IsPrimitiveEquals(prim, prim::kPrimMirror)) {
    return ad_param_->anfnode_to_variable_adjoint_.at(cnode->input(kIndex1))->k_node();
  } else {
    auto c_k_node = ad_param_->tape_->FuncGraph::NewCNode(cnode_inputs);
    c_k_node->set_abstract(cnode->abstract());
    // In jit, copy forward graph cnode info to bprop graph
    if (jit_by_value && cnode->forward().first != nullptr) {
      auto new_v_node = PyNativeAlgo::Common::CreateValueNodeByValue(cnode->forward().first->value(),
                                                                     cnode->forward().first->abstract());
      c_k_node->set_forward(new_v_node, cnode->forward().second);
      ad_param_->tape_->set_used_forward_nodes({c_k_node});
    }
    c_k_node->AddAttr(bprop_pass::kIsKNode, MakeValue(true));
    return c_k_node;
  }
}

void IrBprop::UpdateNextEdgeForDict(const IrFunctionNodePtr &fn, const AnfNodePtr &din, const ValuePtr &input_arg,
                                    const AbstractBasePtr &abs) {
  auto value_dict = input_arg->cast<ValueDictionaryPtr>()->value();
  const auto &abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
  MS_EXCEPTION_IF_NULL(abs_dict);
  if (value_dict.size() != abs_dict->size()) {
    MS_LOG(EXCEPTION) << "Get value dict size " << value_dict.size() << " not equal to abstract size "
                      << abs_dict->size();
  }
  for (size_t i = 0; i < value_dict.size(); ++i) {
    auto sub_value = value_dict[i];
    auto key_item = PyNativeAlgo::Common::CreateValueNodeByValue(sub_value.first, abs_dict->elements()[i].first);
    CNodePtr new_din = ad_param_->tape_->FuncGraph::NewCNode({NewValueNode(prim::kPrimDictGetItem), din, key_item});
    new_din->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(abs_dict->elements()[i].second));
    if (din == fn->fake_dout()) {
      // The new_din's index input is fn->fake_dout()
      LazyAddUser(fn->fake_dout(), new_din, 1);
    }
    // Add next edge to fn
    UpdateNextEdge(fn, new_din, sub_value.second, abs_dict->elements()[i].second);
  }
}

void IrBprop::UpdateNextEdge(const IrFunctionNodePtr &fn, const AnfNodePtr &din, const ValuePtr &input_arg,
                             const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(din);
  MS_EXCEPTION_IF_NULL(input_arg);
  if (input_arg->isa<tensor::BaseTensor>()) {
    tensor::BaseTensorPtr input_tensor = nullptr;
    input_tensor = input_arg->cast<tensor::BaseTensorPtr>();
    auto auto_grad_meta_data = input_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto variable = auto_grad_meta_data->variable();
    if (variable == nullptr || !variable->is_need_grad()) {
      return;
    }
    auto real_din = HandleRealToComplex(input_tensor, abs, din, fn->tape());
    auto new_din = TraceInput(fn, variable->out_value(), variable->ir_function_node()->accumulate_dout()->abstract(),
                              input_tensor, real_din);
    fn->AddNextEdge(variable, new_din);
  } else if (input_arg->isa<ValueSequence>()) {
    auto value_seq = input_arg->cast<ValueSequencePtr>()->value();
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (value_seq.size() != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Get value sequence size " << value_seq.size() << " not equal to abstract size "
                        << abs_seq->size();
    }
    for (size_t i = 0; i < value_seq.size(); ++i) {
      auto sub_value = value_seq[i];
      CNodePtr new_din = ad_param_->tape_->FuncGraph::NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), din, NewValueNode(SizeToLong(i))});
      new_din->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(abs_seq->elements()[i]));
      if (din == fn->fake_dout()) {
        // The new_din's index input is fn->fake_dout()
        LazyAddUser(fn->fake_dout(), new_din, 1);
      }
      // Add next edge to fn
      UpdateNextEdge(fn, new_din, sub_value, abs_seq->elements()[i]);
    }
  } else if (input_arg->isa<tensor::COOTensor>()) {
    auto input_tensor = input_arg->cast<tensor::COOTensorPtr>()->GetIndices();
    UpdateNextEdge(fn, din, input_tensor, PyNativeAlgo::Common::SetAbstractValueToAnyValue(input_tensor->ToAbstract()));
  } else if (input_arg->isa<tensor::CSRTensor>()) {
    auto input_tensor = input_arg->cast<tensor::CSRTensorPtr>()->GetIndices();
    UpdateNextEdge(fn, din, input_tensor, PyNativeAlgo::Common::SetAbstractValueToAnyValue(input_tensor->ToAbstract()));
  } else if (input_arg->isa<ValueDictionary>()) {
    UpdateNextEdgeForDict(fn, din, input_arg, abs);
  } else {
    MS_LOG(DEBUG) << "It is not tensor, not need derivation " << input_arg->ToString();
    return;
  }
}

AnfNodePtr IrBprop::TraceInput(const IrFunctionNodePtr &fn, const ValuePtr &out_value,
                               const abstract::AbstractBasePtr &out_abs, const tensor::BaseTensorPtr &input_tensor,
                               const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(out_value);
  MS_EXCEPTION_IF_NULL(out_abs);
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(din);

  // The node corresponding output tensor is the same as the currently used tensor
  if (out_value->isa<tensor::BaseTensor>()) {
    // out_value is be used, may be it is one of multiple output
    auto out_tensor = out_value->cast<tensor::BaseTensorPtr>();
    if (input_tensor->id() == out_tensor->id()) {
      return din;
    }
    return PyNativeAlgo::AutoGrad::BuildSpecialNode(ad_param_->tape_, out_value, out_abs, SpecialType::kZerosLikeType);
  } else if (out_value->isa<ValueSequence>()) {
    // The corresponding output of node is ValueSequence, but used one of it
    AnfNodePtrList inputs;
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    auto value_seq = out_value->cast<ValueSequencePtr>();
    auto abs_seq = out_abs->cast<abstract::AbstractSequencePtr>();
    if (abs_seq == nullptr) {
      MS_LOG(EXCEPTION) << "Get output abstract " << out_abs->ToString() << ", not abstract sequence";
    }
    int index = -1;
    for (size_t i = 0; i < value_seq->size(); ++i) {
      // Find the value's din, if value equal to sub_value, means value be used, is it will get din; Otherwise value's
      // din is zero , which set by second branch condition above
      auto new_din = TraceInput(fn, value_seq->value()[i], abs_seq->elements()[i], input_tensor, din);
      (void)inputs.emplace_back(new_din);

      // if exist din == fake_dout, we record it in user vector
      if (din == fn->fake_dout() && new_din == din) {
        index = static_cast<int>(inputs.size()) - 1;
      }
    }
    auto new_din = ad_param_->tape_->FuncGraph::NewCNode(inputs);
    new_din->set_abstract(out_abs);
    if (index != -1) {
      LazyAddUser(fn->fake_dout(), new_din, index);
    }
    return new_din;
  } else if (out_value->isa<ValueDictionary>()) {
    return TraceInputForDict(fn, out_value, out_abs, input_tensor, din);
  }
  MS_LOG(DEBUG) << "Get non tensor input " << out_value->ToString();
  return PyNativeAlgo::AutoGrad::BuildSpecialNode(ad_param_->tape_, out_value, out_abs, SpecialType::kZerosLikeType);
}

AnfNodePtr IrBprop::TraceInputForDict(const IrFunctionNodePtr &fn, const ValuePtr &out_value,
                                      const abstract::AbstractBasePtr &out_abs,
                                      const tensor::BaseTensorPtr &input_tensor, const AnfNodePtr &din) {
  // The corresponding output of node is ValueDictionary, but used one of it
  AnfNodePtrList key_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  AnfNodePtrList value_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  abstract::AbstractBasePtrList local_key_abs_inputs;
  abstract::AbstractBasePtrList local_value_abs_inputs;
  auto value_dict = out_value->cast<ValueDictionaryPtr>();
  auto abs_dict = out_abs->cast<abstract::AbstractDictionaryPtr>();
  MS_EXCEPTION_IF_NULL(abs_dict);
  int index = -1;
  for (size_t i = 0; i < value_dict->size(); ++i) {
    // Find the value's din, if value equal to sub_value, means value be used, is it will get din; Otherwise value's
    // din is zero, which set by second branch condition above
    (void)key_inputs.emplace_back(
      PyNativeAlgo::Common::CreateValueNodeByValue(value_dict->value()[i].first, abs_dict->elements()[i].first));
    (void)local_key_abs_inputs.emplace_back(abs_dict->elements()[i].first);
    auto new_din = TraceInput(fn, value_dict->value()[i].second, abs_dict->elements()[i].second, input_tensor, din);
    (void)value_inputs.emplace_back(new_din);
    (void)local_value_abs_inputs.emplace_back(abs_dict->elements()[i].second);

    // if exist din == fake_dout, we record it in user vector
    if (din == fn->fake_dout() && new_din == din) {
      index = static_cast<int>(value_inputs.size()) - 1;
    }
  }
  auto local_key_node = ad_param_->tape_->NewCNode(key_inputs);
  local_key_node->set_abstract(std::make_shared<abstract::AbstractTuple>(local_key_abs_inputs));
  auto local_value_node = ad_param_->tape_->NewCNode(value_inputs);
  local_value_node->set_abstract(std::make_shared<abstract::AbstractTuple>(local_value_abs_inputs));
  auto new_din = ad_param_->tape_->NewCNode({NewValueNode(prim::kPrimMakeDict), local_key_node, local_value_node});
  new_din->set_abstract(abs_dict);
  if (index != -1) {
    LazyAddUser(fn->fake_dout(), new_din, index);
  }
  return new_din;
}

void IrBprop::AddTupleGetItemUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  (void)ad_param_->users_.tuple_getitem_user_[node].emplace_back(user, index);
}

void IrBprop::UpdateLazyUser() {
  // For lazy add user data, we need emplace to user.
  for (const auto &user_data : ad_param_->lazy_user_data_) {
    AddUser(std::get<kIndex0>(user_data), std::get<kIndex1>(user_data), std::get<kIndex2>(user_data));
  }
}

void IrBprop::LazyAddUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(user);
  (void)ad_param_->lazy_user_data_.emplace_back(node, user, index);
}
}  // namespace mindspore::pynative::autograd
