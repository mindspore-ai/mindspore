/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/pi/graph_capture/graph_build_helper.h"

#include <string>
#include <utility>
#include <memory>
#include <algorithm>

#include "ir/func_graph_flag.h"
#include "ir/func_graph_cloner.h"
#include "primitive/framework_ops.h"
#include "frontend/jit/pi/external.h"
#include "frontend/jit/pi/graph_capture/graph_build.h"
#include "frontend/jit/pi/graph_build/build_graph_utils.h"
#include "frontend/jit/ps/action.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "frontend/jit/ps/static_analysis/static_analysis.h"
#include "frontend/jit/pi/graph_capture/abstract_wrapper.h"
#include "frontend/jit/pi/graph_build/func_graph_builder.h"
#include "include/frontend/operator/composite/grad_operation.h"

namespace mindspore {
namespace pijit {
GraphBuildHelperPtr GraphBuildHelperFactory(const py::object &object) {
  auto value = ConvertPyObjToValue(object);
  if (value == nullptr) {
    return nullptr;
  }
  // todo: use object to determine helper to build later.
  const std::string &callable_str = value->ToString();
  const std::string &grad_prefix = "MetaFuncGraph-grad";
  if (callable_str.substr(0, grad_prefix.size()) == grad_prefix) {
    return std::make_shared<GradGraphBuildHelper>();
  }
  return nullptr;
}

GraphBuildHelperPtr GetCallNodeGraphBuildHelper(CallNode *call_node) {
  auto callable_node = call_node->input(0);
  MS_EXCEPTION_IF_NULL(callable_node);
  auto abstract_wrapper = callable_node->abstract_wrapper();
  if (abstract_wrapper == nullptr) {
    return nullptr;
  }
  return abstract_wrapper->graph_builder_helper();
}

AbstractWrapperPtr GraphBuildHelper::Prepare(GraphBuilder *graph_builder, const CallInfo &call_info) {
  AddCallInfo(call_info);
  return PrepareInner(graph_builder, call_info);
}

void GraphBuildHelper::AddCallInfo(const CallInfo &call_info) { call_info_list_.push_back(call_info); }

void GraphBuildHelper::CheckCallInfoListSize(size_t index) {
  if (call_info_list_.size() <= index) {
    MS_LOG(INTERNAL_EXCEPTION) << "Index " << index << " out of range for call_info_list_ with length"
                               << call_info_list_.size();
  }
}

ValuePtr GraphBuildHelper::GetValue(size_t index) {
  CheckCallInfoListSize(index);
  return call_info_list_[index].value;
}

py::object GraphBuildHelper::GetObject(size_t index) {
  CheckCallInfoListSize(index);
  return call_info_list_[index].object;
}

AbstractWrapperPtrList GraphBuildHelper::GetInputsAbstractWrapper(size_t index) {
  CheckCallInfoListSize(index);
  return call_info_list_[index].inputs_abstract_wrapper;
}

AbstractWrapperPtr GradGraphBuildHelper::PrepareInner(GraphBuilder *graph_builder, const CallInfo &call_info) {
  auto func_graph_builder = graph_builder->FGBuilder();
  MS_EXCEPTION_IF_NULL(func_graph_builder);
  const std::string grad_prefix = "MetaFuncGraph-grad";
  const std::string fake_node_key_prefix = "FakeNodeKey";
  std::vector<AnfNodePtr> input_node_list;

  const auto &callable_value = call_info.value;
  const auto &inputs_abstract_wrapper = call_info.inputs_abstract_wrapper;
  (void)input_node_list.emplace_back(NewValueNode(callable_value));
  std::stringstream ss;
  for (const auto &input_wrapper : inputs_abstract_wrapper) {
    auto node = func_graph_builder->FindOrCreateNodeByWrapper(input_wrapper);
    if (node == nullptr) {
      // When build grad operation node failed, let forward net run pi jit.
      constexpr size_t forward_net_index = 0;
      auto forward_net_object = AbstractWrapper::FetchPythonObject(inputs_abstract_wrapper[forward_net_index]);
      (void)AbstractWrapper::MarkObjectPiJItShouldCompile(forward_net_object);
      return nullptr;
    }
    ss << input_wrapper->ToString();
    (void)input_node_list.emplace_back(node);
  }

  // todo: need to used real function object later.
  auto output_py_obj = py::str(fake_node_key_prefix + " " + grad_prefix + " " + ss.str());
  auto abs = abstract::ToAbstract(MakeValue(ConvertPyObjToValue(output_py_obj)));
  auto abstract_wrapper = func_graph_builder->AddNodeWithAbstract(input_node_list, abs);
  const GradGraphBuildHelperPtr &helper = std::make_shared<GradGraphBuildHelper>(*this);
  abstract_wrapper->set_graph_builder_helper(helper);
  return abstract_wrapper;
}

AbstractWrapperPtr GradGraphBuildHelper::Build(GraphBuilder *graph_builder, CallNode *call_node) {
  MS_LOG(DEBUG) << "Start build graph for grad operation";
  auto grad_net_node = static_cast<CallNode *>(call_node->input(0));
  if (grad_net_node == nullptr) {
    return nullptr;
  }
  auto grad_net_wrapper = grad_net_node->abstract_wrapper();
  if (grad_net_wrapper == nullptr) {
    MS_LOG(ERROR) << "Fail to get abstract wrapper for grad net node: " << grad_net_node->ToString();
    return nullptr;
  }
  constexpr size_t grad_operation_index = 0;
  constexpr size_t forward_net_index = 1;
  auto graph = graph_builder->GetGraph();
  bool guard_grad_operation = graph->GuardValueNode(grad_net_node->input(grad_operation_index), GId);
  if (!guard_grad_operation) {
    MS_LOG(WARNING) << "Guard GradOperation value node failed, value node: "
                    << grad_net_node->input(grad_operation_index)->ToString();
  }

  auto forward_net_object = grad_net_node->input(forward_net_index)->GetVobj()->GetPyObject();
  (void)pi_jit_should_compile(forward_net_object);

  bool guard_forward_net = graph->GuardValueNode(grad_net_node->input(forward_net_index), GId);
  if (!guard_forward_net) {
    MS_LOG(WARNING) << "Guard forward net value node for GradOperation failed, value node: "
                    << grad_net_node->input(forward_net_index)->ToString();
  }

  auto forward_result = BuildForwardGraph(graph_builder, call_node);
  auto forward_fg = forward_result.first;
  if (forward_fg == nullptr) {
    MS_LOG(INFO) << "Build forward fg failed.";
    return nullptr;
  }
  if (py::isinstance<Cell>(forward_net_object)) {
    HandleCustomBProp(forward_fg, forward_net_object);
  }

  const auto &inputs_wrapper = HandleInputsForGrad(graph_builder, call_node, forward_result.second);
  for (auto input_wrapper : inputs_wrapper) {
    if (input_wrapper == nullptr) {
      MS_LOG(EXCEPTION) << "Input wrapper is NULL, failed to build graph.";
    }
    MS_LOG(INFO) << "input wrapper is: " << input_wrapper->ToString();
  }
  auto ret = BuildGradNode(graph_builder->FGBuilder(), grad_net_wrapper, forward_fg, inputs_wrapper);
  if (ret == nullptr) {
    return nullptr;
  }
  HandleGradForwardSideEffect(graph_builder, forward_fg, ret, graph_builder->get_prev_call_builder(), call_node);
  return ret;
}

AbstractWrapperPtr GradGraphBuildHelper::BuildGradNode(const FuncGraphBuilderPtr &func_graph_builder,
                                                       const AbstractWrapperPtr &key, const FuncGraphPtr &forward_fg,
                                                       const AbstractWrapperPtrList &inputs) {
  AbstractWrapperPtr ret = nullptr;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    ret = HandleGrad(func_graph_builder, key, forward_fg, inputs);
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Failed to build grad node with key: " << key << ". The exception:\n" << e.what();
  }
  return ret;
}

// For GradOperation(net, ...)(forward_inputs), two nodes should be evaluated together as a graph.
// Before:
//   fake_node: GradOperation(net, other_inputs)
// After:
//   fg(other_inputs, forward_inputs)
//     grad_net_node:    DoSignature(GradOperation)(net, other_inputs)
//     grad_result_node: grad_net_node(forward_inputs) or unpack_call(grad_net_node, forward_inputs)
//     return grad_result_node
//   final node for evaluated: fg(other_inputs, forward_inputs)
AbstractWrapperPtr GradGraphBuildHelper::HandleGrad(const FuncGraphBuilderPtr &func_graph_builder,
                                                    const AbstractWrapperPtr &key, const FuncGraphPtr &forward_fg,
                                                    const AbstractWrapperPtrList &inputs) {
  auto fake_node = func_graph_builder->ReadLocalVariable(key);
  if (fake_node == nullptr || !fake_node->isa<CNode>()) {
    MS_LOG(INFO) << "Failed to find corresponding fake GradOperation node for key: " << key;
    return nullptr;
  }
  auto fake_node_abstract = fake_node->abstract();
  if (fake_node_abstract == nullptr) {
    MS_LOG(INFO) << "When handling grad, fail to find abstract for fake node: " << fake_node->DebugString();
    return nullptr;
  }

  const auto &pre_wrapper = GetInputsAbstractWrapper(0);
  std::vector<AnfNodePtr> fake_node_inputs;
  for (auto e : pre_wrapper) {
    auto cur_node = func_graph_builder->FindOrCreateNodeByWrapper(e);
    MS_EXCEPTION_IF_NULL(cur_node);
    fake_node_inputs.push_back(cur_node);
  }

  const auto &meta_object = GetObject(0);
  auto value = ConvertPyCallableToValue(meta_object);
  MS_EXCEPTION_IF_NULL(value);
  auto meta = value->cast<MetaFuncGraphPtr>();
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(forward_fg);
  auto origin_forward_fg_output = forward_fg->output();
  auto fake_cnode = fake_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(fake_cnode);
  auto meta_node = NewValueNode(std::make_shared<prim::DoSignaturePrimitive>(meta->name(), meta));
  std::vector<AnfNodePtr> grad_net_node_inputs{meta_node, NewValueNode(forward_fg)};
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  for (size_t i = 1; i < fake_node_inputs.size(); ++i) {
    auto para = AddParameter(fg);
    (void)grad_net_node_inputs.emplace_back(para);
  }
  auto grad_net_node = fg->NewCNodeInOrder(grad_net_node_inputs);
  std::vector<AnfNodePtr> grad_result_node_inputs;
  grad_result_node_inputs.push_back(grad_net_node);
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto para = AddParameter(fg);
    (void)grad_result_node_inputs.emplace_back(para);
  }
  auto grad_result_node = fg->NewCNodeInOrder(grad_result_node_inputs);
  fg->set_output(grad_result_node);
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    DumpIR("pijit_grad_fg.ir", fg);
  }
  std::vector<AnfNodePtr> final_node_input = {NewValueNode(fg)};
  std::vector<AbstractBasePtr> final_node_abs;
  for (size_t i = 1; i < fake_node_inputs.size(); ++i) {
    AnfNodePtr cur_input = fake_node_inputs[i];
    MS_EXCEPTION_IF_NULL(cur_input);
    auto cur_input_abs = cur_input->abstract();
    MS_EXCEPTION_IF_NULL(cur_input_abs);
    final_node_input.push_back(cur_input);
    final_node_abs.push_back(cur_input_abs);
  }
  for (auto input_wrapper : inputs) {
    auto node = func_graph_builder->FindOrCreateNodeByWrapper(input_wrapper);
    MS_EXCEPTION_IF_NULL(node);
    (void)final_node_input.emplace_back(node);
    (void)final_node_abs.emplace_back(node->abstract());
  }
  for (auto abs : final_node_abs) {
    MS_LOG(INFO) << "final input abstract: " << abs->ToString();
  }
  fg->set_manager(func_graph_builder->manager());
  auto analyze_res = pipeline::AbstractAnalyze(fg, final_node_abs);
  MS_EXCEPTION_IF_NULL(analyze_res.eval_result);
  auto final_abs = analyze_res.eval_result->abstract();
  MS_EXCEPTION_IF_NULL(final_abs);
  auto abstract_wrapper = func_graph_builder->AddNodeWithAbstract(final_node_input, final_abs);
  auto cur_forward_fg_output = forward_fg->output();
  if (origin_forward_fg_output != cur_forward_fg_output) {
    // has_aux for GradOperation will change the output of forward fg.
    forward_fg->set_output(origin_forward_fg_output);
  }
  UpdateGradInfo(meta);
  MS_LOG(INFO) << "Build grad success with abstract wrapper: " << abstract_wrapper->ToString();
  return abstract_wrapper;
}

std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>> GradGraphBuildHelper::BuildForwardGraph(
  GraphBuilder *graph_builder, CallNode *call_node) {
  MS_LOG(DEBUG) << "Start build forward graph";
  auto grad_net_node = static_cast<CallNode *>(call_node->input(0));
  MS_EXCEPTION_IF_NULL(grad_net_node);
  constexpr size_t grad_operation_index = 0;
  constexpr size_t forward_node_index = 1;
  auto grad_operation_node = grad_net_node->input(grad_operation_index);
  auto grad_object = grad_operation_node->GetVobj()->GetPyObject();
  auto forward_node = grad_net_node->input(forward_node_index);
  bool has_sense = py::hasattr(grad_object, "sens_param") && (grad_object.attr("sens_param").ptr() == Py_True);
  auto func_info = forward_node->GetVobj()->GetPyObject();
  bool is_cell = py::isinstance<Cell>(func_info);
  MS_EXCEPTION_IF_NULL(func_info.ptr());

  auto self_node = is_cell ? forward_node : nullptr;
  BindArgumentsHelper<ValueNode *> bind_helper = graph_builder->PackInputsForFunc(
    func_info, call_node->GetOpcode(), call_node->inputs(), call_node->kw_names().ptr(), self_node, has_sense);

  auto bind_arguments_result = bind_helper.results();
  const auto &bind_args = bind_arguments_result.args_;
  const auto &bind_vargs = bind_arguments_result.va_;
  const auto &bind_kwargs = bind_arguments_result.kw_va_;

  func_info = GraphBuilder::FindPyFunc(AObject::Convert(func_info));
  graph_builder->DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(func_info)});
  int arg_size = 0;
  for (auto arg : bind_args) {
    graph_builder->push(arg);
    arg_size = arg_size + 1;
  }
  for (auto varg : bind_vargs) {
    graph_builder->push(varg);
    arg_size = arg_size + 1;
  }

  if (!bind_kwargs.empty()) {
    // Use CALL_FUNCTION_KW to build forward node.
    MS_LOG(EXCEPTION) << "Do not handle kwargs yet.";
  } else {
    MS_LOG(DEBUG) << "Start trace bytecodes of forward graph";
    auto call_instr = graph_builder->NewCallFuncInstr(arg_size);
    graph_builder->DoCall(call_instr);
  }
  auto forward_call_value_node = graph_builder->pop();
  // This forward node is only used for abstract, no need to attach on graph.
  auto cur_graph_builder = graph_builder->FGBuilder();
  MS_EXCEPTION_IF_NULL(cur_graph_builder);
  auto forward_call_wrapper = forward_call_value_node->abstract_wrapper();
  if (forward_call_wrapper == nullptr) {
    MS_LOG(INFO) << "Failed to get wrapper of forward graph.";
    return std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>>(nullptr, bind_helper);
  }
  auto forward_call_node = cur_graph_builder->ReadLocalVariable(forward_call_wrapper);
  MS_EXCEPTION_IF_NULL(forward_call_node);
  cur_graph_builder->EraseCandidateIsolatedNode(forward_call_node);

  auto forward_graph_builder = graph_builder->get_prev_call_builder();
  if (forward_graph_builder == nullptr) {
    MS_LOG(INFO) << "Failed to get function graph builder for forward graph.";
    return std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>>(nullptr, bind_helper);
  }
  auto fg = forward_graph_builder->FGBuilder()->graph();
  if (fg == nullptr) {
    MS_LOG(INFO) << "Failed to get function graph builder for forward graph.";
    return std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>>(nullptr, bind_helper);
  }
  fg = BasicClone(fg);
  if (bind_vargs.size() != 0) {
    MS_LOG(INFO) << "Build call graph for forward graph.";
    std::vector<size_t> arg_len = {bind_args.size(), bind_vargs.size(), bind_kwargs.size()};
    auto outer_fg = BuildCallForwardGraphForGrad(fg, arg_len, is_cell);
    return std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>>(outer_fg, bind_helper);
  }

  return std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>>(fg, bind_helper);
}

FuncGraphPtr GradGraphBuildHelper::BuildCallForwardGraphForGrad(const FuncGraphPtr &fg,
                                                                const std::vector<size_t> &arg_len, bool is_cell) {
  MS_LOG(INFO) << "Build outer fg for vargs scene.";
  auto origin_forward_abs = fg->output()->abstract();
  MS_EXCEPTION_IF_NULL(origin_forward_abs);
  MS_LOG(INFO) << "origin forward abs: " << origin_forward_abs->ToString();

  AnfNodePtrList call_forward_inputs = {NewValueNode(fg)};
  auto outer_fg = std::make_shared<FuncGraph>();
  constexpr auto args_index = 0;
  constexpr auto vargs_index = 1;
  // Eliminate self input for cell when building grad graph.
  size_t input_offset = is_cell ? 1 : 0;
  for (size_t i = 0 + input_offset; i < arg_len[args_index]; ++i) {
    auto para = AddParameter(outer_fg);
    (void)call_forward_inputs.emplace_back(para);
  }
  if (arg_len[vargs_index] != 0) {
    AnfNodePtrList vargs_tuple = {NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < arg_len[vargs_index]; ++i) {
      auto para = AddParameter(outer_fg);
      (void)vargs_tuple.emplace_back(para);
    }
    auto vargs_node = outer_fg->NewCNodeInOrder(vargs_tuple);
    (void)call_forward_inputs.emplace_back(vargs_node);
  }
  // This is a tmp way to fix empty kwargs.
  if (fg->parameters().size() == call_forward_inputs.size()) {
    (void)call_forward_inputs.emplace_back(NewValueNode(0));
  }
  auto call_forward_node = outer_fg->NewCNodeInOrder(call_forward_inputs);
  call_forward_node->set_abstract(origin_forward_abs);
  outer_fg->set_output(call_forward_node);
  return outer_fg;
}

void GradGraphBuildHelper::HandleCustomBProp(const FuncGraphPtr &graph, const py::object &obj) const {
  if (graph == nullptr || obj.ptr() == nullptr) {
    return;
  }
  if (!py::hasattr(obj, parse::CUSTOM_BPROP_NAME)) {
    return;
  }
  bool enable_bprop_debug = py::cast<bool>(py::getattr(obj, "bprop_debug"));
  FuncGraphPtr bprop_graph = enable_bprop_debug
                               ? parse::ConvertToBpropCut(obj)
                               : parse::ConvertToFuncGraph(obj, {}, parse::PYTHON_MOD_GET_BPROP_METHOD);
  if (bprop_graph != nullptr) {
    (void)graph->transforms().emplace(parse::CUSTOM_BPROP_NAME, FuncGraphTransform(bprop_graph));
    (void)bprop_graph->transforms().emplace("primal", FuncGraphTransform(graph));
    graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
    graph->set_flag(FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP, true);
    MS_LOG(INFO) << "Add custom bprop to graph.";
  }
  return;
}

void GradGraphBuildHelper::HandleGradForwardSideEffect(GraphBuilder *graph_builder, const FuncGraphPtr &forward_fg,
                                                       const AbstractWrapperPtr &grad,
                                                       const GraphBuilderPtr &subgraph_builder, CallNode *call_node) {
  const auto &side_effect_outputs = subgraph_builder->side_effect_outputs();
  if (side_effect_outputs.empty() || !grad_info_.get_value_) {
    return;
  }
  // For value_and_grad with forward net side effect,
  // the output is format ((real_forward, side_effect_output), real_grad)
  // Need to adjust abstract wrapper for call node to (real_forward, real_grad) and adjust side_effect_outputs as well.
  MS_EXCEPTION_IF_NULL(forward_fg->output());
  auto grad_abstract = grad->abstract();
  MS_EXCEPTION_IF_NULL(grad_abstract);
  if (!grad_abstract->isa<abstract::AbstractTuple>()) {
    return;
  }

  AbstractWrapperPtr forward_idx = graph_builder->FGBuilder()->AddLocalVariable(py::int_(0));
  auto forward = graph_builder->FGBuilder()->AddNode(prim::kPrimTupleGetItem, {grad, forward_idx});
  MS_EXCEPTION_IF_NULL(forward);
  AbstractWrapperPtr grad_idx = graph_builder->FGBuilder()->AddLocalVariable(py::int_(1));
  auto real_grad = graph_builder->FGBuilder()->AddNode(prim::kPrimTupleGetItem, {grad, grad_idx});
  MS_EXCEPTION_IF_NULL(real_grad);
  AbstractWrapperPtr real_forward_idx = graph_builder->FGBuilder()->AddLocalVariable(py::int_(0));
  auto real_forward = graph_builder->FGBuilder()->AddNode(prim::kPrimTupleGetItem, {forward, real_forward_idx});
  MS_EXCEPTION_IF_NULL(real_forward);
  for (size_t i = 0; i < side_effect_outputs.size(); ++i) {
    AbstractWrapperPtr cur_idx = graph_builder->FGBuilder()->AddLocalVariable(py::int_(i + 1));
    auto cur_side_effect_output = graph_builder->FGBuilder()->AddNode(prim::kPrimTupleGetItem, {forward, cur_idx});
    MS_EXCEPTION_IF_NULL(cur_side_effect_output);
    side_effect_outputs[i]->set_abstract_wrapper(cur_side_effect_output);
  }
  auto real_ret = graph_builder->FGBuilder()->AddNode(prim::kPrimMakeTuple, {real_forward, real_grad});
  MS_EXCEPTION_IF_NULL(real_ret);
  call_node->SetVobj(AObject::Convert(real_ret));
  call_node->set_abstract_wrapper(real_ret);
}

AbstractWrapperPtrList GradGraphBuildHelper::HandleInputsForGrad(GraphBuilder *graph_builder, CallNode *call_node,
                                                                 BindArgumentsHelper<ValueNode *> forward_inputs) {
  MS_LOG(DEBUG) << "Handle inputs for grad op";
  auto grad_net_node = static_cast<CallNode *>(call_node->input(0));
  MS_EXCEPTION_IF_NULL(grad_net_node);
  constexpr size_t grad_operation_index = 0;
  constexpr size_t forward_node_index = 1;
  auto grad_operation_node = grad_net_node->input(grad_operation_index);
  auto grad_object = grad_operation_node->GetVobj()->GetPyObject();
  auto forward_node = grad_net_node->input(forward_node_index);
  bool has_sense = py::hasattr(grad_object, "sens_param") && (grad_object.attr("sens_param").ptr() == Py_True);
  auto func_info = forward_node->GetVobj()->GetPyObject();
  MS_EXCEPTION_IF_NULL(func_info.ptr());

  const auto &bind_arguments_result = forward_inputs.results();
  const auto &bind_args = bind_arguments_result.args_;
  const auto &bind_vargs = bind_arguments_result.va_;
  const auto &bind_kwargs = bind_arguments_result.kw_va_;

  auto wrapper_args = graph_builder->HandleInputArgs(bind_args);
  const auto &wrapper_vargs = graph_builder->HandleInputArgs(bind_vargs);

  bool get_all = py::hasattr(grad_object, "get_all") && (grad_object.attr("get_all").ptr() == Py_True);
  bool get_by_list = py::hasattr(grad_object, "get_by_list") && (grad_object.attr("get_by_list").ptr() == Py_True);
  auto offset = py::isinstance<Cell>(func_info) ? 1 : 0;
  std::vector<ValueNode *> forward_input(bind_args.begin() + offset, bind_args.end());
  size_t input_grad_cnt = get_all ? forward_input.size() : (get_by_list ? 0 : (forward_input.empty() ? 0 : 1));

  for (size_t index = 0; index < input_grad_cnt; index++) {
    if (!forward_input[index]->has_abstract_wrapper() || forward_input[index]->GetVobj() == nullptr) {
      continue;
    }
    auto obj = forward_input[index]->GetVobj()->GetPyObject();
    if (py::isinstance<py::none>(obj)) {
      continue;
    }
    auto wrapper = forward_input[index]->abstract_wrapper();
    auto node = graph_builder->FGBuilder()->FindNodeByWrapper(wrapper);
  }

  if (has_sense) {
    if (!bind_vargs.empty() || !bind_kwargs.empty()) {
      MS_LOG(EXCEPTION) << "Do not support sense param with vargs and kwargs yet.";
    }
    MS_EXCEPTION_IF_CHECK_FAIL(call_node->inputs().size() > bind_args.size(), "Arg size check failed.");
    auto sens_value_node = call_node->inputs().back();
    auto new_wrapper = sens_value_node->has_abstract_wrapper()
                         ? sens_value_node->abstract_wrapper()
                         : graph_builder->FGBuilder()->AddLocalVariable(sens_value_node->GetVobj()->GetPyObject());
    sens_value_node->set_abstract_wrapper(new_wrapper);
    wrapper_args.push_back(new_wrapper);
  }

  AbstractWrapperPtrList final_wrapper;
  // Eliminate self input for cell when building grad graph.
  bool input_offset = py::isinstance<Cell>(func_info) ? 1 : 0;
  (void)std::copy(wrapper_args.begin() + input_offset, wrapper_args.end(), std::back_inserter(final_wrapper));
  (void)std::copy(wrapper_vargs.begin(), wrapper_vargs.end(), std::back_inserter(final_wrapper));
  return final_wrapper;
}

void GradGraphBuildHelper::UpdateGradInfo(const ValuePtr &meta) {
  MS_EXCEPTION_IF_NULL(meta);
  auto grad = meta->cast<prim::GradOperationPtr>();
  MS_EXCEPTION_IF_NULL(grad);
  grad_info_.get_all_ = grad->get_all_;
  grad_info_.get_by_list_ = grad->get_by_list_;
  grad_info_.sens_param_ = grad->sens_param_;
  grad_info_.get_by_position_ = grad->get_by_position_;
  grad_info_.has_aux_ = grad->has_aux_;
  grad_info_.get_value_ = grad->get_value_;
  grad_info_.return_ids_ = grad->return_ids_;
  grad_info_.merge_forward_ = grad->merge_forward_;
}
}  // namespace pijit
}  // namespace mindspore
