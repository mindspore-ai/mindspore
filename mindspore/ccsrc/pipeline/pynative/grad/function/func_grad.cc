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

#include "pipeline/pynative/grad/function/func_grad.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/primitive_utils.h"
#include "include/common/utils/hook.h"
#include "pipeline/pynative/pynative_utils.h"
#include "ops/framework_ops.h"
#include "ops/other_ops.h"

namespace mindspore::pynative::autograd {
namespace {
ValuePtr Add(const ValuePtr &input, const ValuePtr &other, const FuncBuilderPtr &func_impl) {
  if (input->isa<None>()) {
    MS_EXCEPTION_IF_NULL(other);
    return other;
  }
  if (other->isa<None>()) {
    MS_EXCEPTION_IF_NULL(input);
    return input;
  }
  auto result = func_impl->Add(input, other);
  MS_EXCEPTION_IF_NULL(result);
  return result;
}

void Add(const ValuePtr &other, size_t input_index, const FuncBuilderPtr &func_impl, std::vector<ValuePtr> *inputs) {
  if (input_index >= inputs->size()) {
    MS_LOG(EXCEPTION) << "The input index should less than inputs size";
  }

  (*inputs)[input_index] = Add(inputs->at(input_index), other, func_impl);
}

ValuePtrList PaddingGradientInput(const ValuePtr &grad, size_t output_size, size_t input_index) {
  ValuePtrList gradients;
  gradients.reserve(output_size);
  for (size_t i = 0; i < output_size; ++i) {
    if (input_index == i) {
      (void)gradients.emplace_back(grad);
    } else {
      // If gradient is not, we just set kNone, then we lazy update zero gradient by
      // LazeUpdateZeroGradient method
      (void)gradients.emplace_back(kNone);
    }
  }
  return gradients;
}

VectorRef GeneratePythonArgs(const OpGradInfoPtr &op_grad_info) {
  VectorRef args;
  size_t input_size = op_grad_info->input_value.size() - op_grad_info->weight_size;
  for (size_t i = 0; i < input_size; ++i) {
    (void)args.emplace_back(op_grad_info->input_value[i]);
  }
  // If we not need recompute, we save output.
  if (!op_grad_info->is_need_recompute) {
    (void)args.emplace_back(op_grad_info->out_value);
  }
  return args;
}

ValuePtr ValueListToValue(const ValuePtrList &values, const abstract::AbstractBasePtr &abs) {
  if (values.size() == kSizeZero) {
    MS_LOG(EXCEPTION) << "tensors size should not be empty!";
  }
  if (values.size() == kSizeOne && !abs->isa<abstract::AbstractSequence>()) {
    return values[kIndex0];
  }
  return std::make_shared<ValueTuple>(values);
}

bool IsOutputBothEmpty(const ValuePtr &input_grads, const ValuePtr &weight_grads) {
  if (!input_grads->isa<ValueTuple>() || !weight_grads->isa<ValueTuple>()) {
    return false;
  }
  auto input_grads_tuple = input_grads->cast<ValueTuplePtr>();
  auto weight_grads_tuple = weight_grads->cast<ValueTuplePtr>();
  return input_grads_tuple->size() == 0 && weight_grads_tuple->size() == 0;
}

ValuePtr GenerateEmptyTupleValue() {
  std::vector<ValuePtr> value_list;
  auto inputs_value = std::make_shared<ValueTuple>(value_list);
  auto weights_value = std::make_shared<ValueTuple>(value_list);
  std::vector<ValuePtr> tuple_list{inputs_value, weights_value};
  return std::make_shared<ValueTuple>(tuple_list);
}

void SetFlattenTensorGradMetaData(const ValuePtrList &flatten_outs, const VariablePtr &variable) {
  for (size_t i = 0; i < flatten_outs.size(); ++i) {
    if (flatten_outs[i]->isa<tensor::BaseTensor>()) {
      auto tensor = flatten_outs[i]->cast<tensor::BaseTensorPtr>();
      auto auto_grad_meta_data = tensor->auto_grad_meta_data();
      if (auto_grad_meta_data == nullptr) {
        MS_LOG(DEBUG) << "Tensor " << tensor->id() << " has no auto_grad_meta_data";
        auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
        tensor->set_auto_grad_meta_data(auto_grad_meta_data);
      }
      auto_grad_meta_data->set_variable(variable);
      auto_grad_meta_data->set_output_index(i);
    }
  }
}

bool IsValidTensorInput(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  return v->isa<tensor::BaseTensor>() || v->isa<tensor::MetaSparseTensor>();
}

bool IsNeedComputeGrad(const ValuePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  if (input->isa<tensor::BaseTensor>()) {
    const auto &input_tensor = input->cast<tensor::BaseTensorPtr>();
    const auto &auto_grad_meta_data = input_tensor->auto_grad_meta_data();
    if (auto_grad_meta_data == nullptr) {
      return false;
    }
    auto variable = auto_grad_meta_data->variable();
    if (variable != nullptr && variable->is_need_grad()) {
      return true;
    }
  } else if (input->isa<ValueSequence>()) {
    auto seq = input->cast<ValueSequencePtr>();
    if (!seq->value().empty() && !seq->value().front()->isa<tensor::BaseTensor>()) {
      return false;
    }
    return std::any_of(seq->value().begin(), seq->value().end(),
                       [](const ValuePtr &val) { return IsNeedComputeGrad(val); });
  }
  return false;
}

NodePtrList GenerateNodeInputs(const OpGradInfoPtr &op_grad_info, const FuncBuilderPtr &emitter) {
  NodePtrList node_inputs;
  node_inputs.reserve(op_grad_info->input_value.size() + kSizeFive);
  for (size_t i = 0; i < op_grad_info->input_value.size(); ++i) {
    auto func_node = emitter->NewFuncNode(op_grad_info->input_value[i], op_grad_info->input_abs[i],
                                          op_grad_info->input_value_grad_type[i]);
    func_node->set_need_compute_grad_out(IsNeedComputeGrad(op_grad_info->input_value[i]));
    (void)node_inputs.emplace_back(func_node);
  }
  (void)node_inputs.emplace_back(
    emitter->NewFuncNode(op_grad_info->out_value, op_grad_info->out_abs, InputType::kOpOutput));
  return node_inputs;
}

ValuePtrList CallBackwardHooks(const ValuePtr &value, ValuePtrList *grad_in) {
  if (value == nullptr) {
    MS_LOG(DEBUG) << "Get null value";
    return *grad_in;
  }
  MS_EXCEPTION_IF_NULL(grad_in);
  auto tensor = value->cast<tensor::BaseTensorPtr>();
  if (tensor == nullptr) {
    MS_LOG(DEBUG) << "Hook just work on tensor, not support value " << value->ToString();
    return *grad_in;
  }
  const auto &auto_grad_meta = tensor->auto_grad_meta_data();
  if (auto_grad_meta == nullptr || auto_grad_meta->backward_hooks().empty()) {
    MS_LOG(DEBUG) << "Get empty backward hooks for tensor id " << tensor->id();
    return *grad_in;
  }
  if (grad_in->size() != kSizeOne) {
    MS_LOG(EXCEPTION) << "Tensor hook just work on one tensor value, not support value sequence";
  }
  for (const auto &hook : auto_grad_meta->backward_hooks()) {
    MS_LOG(DEBUG) << "Run hook id " << hook.first;
    MS_EXCEPTION_IF_NULL(hook.second);
    (*grad_in)[kIndex0] = (*(hook.second))(grad_in->front());
  }
  MS_LOG(DEBUG) << PyNativeAlgo::Common::PrintDebugInfo(*grad_in, "After hook print gradient in: ");
  return *grad_in;
}

void ReleaseResource(const VariablePtr &variable) {
  const auto &forward = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor();
  if (forward->enable_async()) {
    const auto task = [variable]() { variable->Release(); };
    const auto &bprop_queue = runtime::Pipeline::Get().bprop_stage();
    if (!bprop_queue->Push(new (std::nothrow) BpropTask(std::move(task)))) {
      bprop_queue->CheckException();
    }
  } else {
    variable->Release();
  }
}
}  // namespace

ValuePtrList FuncBackwardNode::CallBackward(const ValuePtrList &gradients_in) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     name(), false);
  MS_LOG(DEBUG) << "Begin CallBackward: " << name();
  PreProcess(gradients_in, emitter_);
  emitter_->SetInputs(name(), &node_inputs_, &attrs_);
  const std::vector<NodePtr> cal_grads_node = grad_func()(emitter_.get());
  ValuePtrList cal_grads_values;
  cal_grads_values.reserve(cal_grads_node.size());
  // Binary op grad result may be nulllptr, we need convert to kNone.
  (void)std::transform(cal_grads_node.begin(), cal_grads_node.end(), std::back_inserter(cal_grads_values),
                       [](const NodePtr &node) -> ValuePtr {
                         if (node == nullptr) {
                           return kNone;
                         }
                         return node->Value();
                       });
  auto gradients = PostProcess(cal_grads_values);
  MS_LOG(DEBUG) << "End CallBackward: " << name();
  return gradients;
}

void FuncBackwardNode::PreProcess(const ValuePtrList &dout, const FuncBuilderPtr &emitter) {
  const size_t output_index = node_inputs_.size() - kIndex1;
  const auto &output_node = node_inputs_[output_index];
  const auto &op_output = output_node->Value();
  if (dout.size() == kSizeOne && !op_output->isa<ValueSequence>()) {
    (void)node_inputs_.emplace_back(emitter->NewFuncNode(dout[kIndex0], output_node->abstract(), InputType::kOpOutput));
  } else {
    (void)node_inputs_.emplace_back(
      emitter->NewFuncNode(std::make_shared<ValueTuple>(dout), output_node->abstract(), InputType::kOpOutput));
  }
}

void FuncBackwardNode::Release() {
  for (const auto &node : node_inputs_) {
    node->SetValue(nullptr);
  }
}

ValuePtrList HookBackwardNode::CallBackward(const ValuePtrList &grads) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     name(), false);
  runtime::Pipeline::Get().WaitForward();
  MS_LOG(DEBUG) << "Begin HookBackwardNode CallBackward ";
  auto gradient = ValueListToValue(grads, out_abstract_);
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  // Python grad func can not process None, we need to convert None to zero tensor.
  auto func_builder = FuncBuilder(name_, device_target, nullptr);
  auto filled_zeros_grad = func_builder.FillZeros(gradient, out_abstract_);
  (void)args_.emplace_back(filled_zeros_grad);
  py::gil_scoped_acquire gil_acquire;
  auto out = prim_->RunHookFunction(args_);
  ValuePtrList gradient_values;
  if (utils::isa<PyObjectRef>(out)) {
    PyObjectRef py_ref = utils::cast<PyObjectRef>(out);
    auto out_py_tuple = py_ref.object_;
    ConvertPyObjectToTensor(out_py_tuple, &gradient_values);
  }
  if (gradient_values.empty()) {
    MS_LOG(EXCEPTION) << "Hook fn output is not <PyObjectRef> type!";
  }
  auto gradient_tensors = PostProcess(gradient_values);
  MS_LOG(DEBUG) << "End HookBackwardNode CallBackward";
  runtime::Pipeline::Get().WaitForward();
  return gradient_tensors;
}

void HookBackwardNode::Release() {
  prim_ = nullptr;
  args_.clear();
}

ValuePtrList GraphBackwardNode::CallBackward(const ValuePtrList &grads) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     name(), false);
  MS_LOG(DEBUG) << "Begin GraphBackwardNode CallBackward ";
  MS_LOG(DEBUG) << PyNativeAlgo::Common::PrintDebugInfo(grads, "bprop cut input grads: ");
  auto graph_call_back = PyNativeAlgo::AutoGrad::CreateGraphCallBack(func_graph_, cache_key_, graph_call_condition_);
  // Add graph din
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto ir_builder = FuncBuilder(name_, device_target, nullptr);
  auto real_dout = LazeUpdateZeroGradient(grads, &ir_builder, op_output_);

  // If output is jit and has dict output. Key and value will converte into tuples for inputs
  if (!graph_call_condition_.jit_out_has_dict_) {
    for (const auto &arg : real_dout) {
      (void)args_.emplace_back(arg);
    }
  } else {
    if (!op_output_->isa<ValueDictionary>()) {
      MS_LOG(EXCEPTION) << "Get wrong data type " << op_output_->ToString();
    }
    const auto &v_dict = op_output_->cast<ValueDictionaryPtr>();
    ValuePtrList key_inputs;
    for (const auto &elem : v_dict->value()) {
      (void)key_inputs.emplace_back(elem.first);
    }
    (void)args_.emplace_back(std::make_shared<ValueTuple>(key_inputs));
    (void)args_.emplace_back(std::make_shared<ValueTuple>(real_dout));
  }
  auto gradient_vec_ref = graph_call_back(args_);
  auto gradient_values = common::AnfAlgo::TransformVectorRefToMultiValue(gradient_vec_ref);
  auto gradient_tensors = PostProcess(gradient_values);
  MS_LOG(DEBUG) << "End GraphBackwardNode CallBackward";
  return gradient_tensors;
}

ValuePtrList GraphRoot::BuildFlattenSensGradient(const ValuePtrList &sens_gradient) const {
  ValuePtrList real_gradients;
  for (const auto &index : gradient_index_) {
    if (index >= sens_gradient.size()) {
      MS_LOG(EXCEPTION) << "Inputs gradient index should smaller than flatten_values size!";
    }
    (void)real_gradients.emplace_back(sens_gradient[index]);
  }
  return real_gradients;
}

FuncGrad::FuncGrad(const ValuePtrList &input_param_values, size_t op_num_in_bprop_graph, bool grad_by_value,
                   bool is_run_recompute) {
  MS_LOG(DEBUG) << "Start FuncGrad, input size: " << input_param_values.size();
  for (size_t i = 0; i < input_param_values.size(); ++i) {
    const auto &input_param_value = input_param_values[i];
    auto func_node = std::make_shared<BackwardNode>("input" + std::to_string(i));
    auto variable = std::make_shared<FuncVariable>(func_node, true);

    if (!input_param_value->isa<ValueSequence>()) {
      // For hook input
      func_node->set_op_output(input_param_value);
      PyNativeAlgo::AutoGrad::SetGradInfoForInputs(input_param_value, variable, &param_meta_grad_info_);
    } else {
      variable->set_is_need_grad(false);
    }
    (void)variable_set_.insert(variable);
    (void)cell_inputs_.emplace_back(input_param_value, variable);
  }
  is_run_recompute_ = is_run_recompute;
  param_meta_grad_info_.reserve(op_num_in_bprop_graph);
  device_target_ = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  func_impl_ = std::make_shared<FuncBuilder>("func_emitter", device_target_);
}

bool FuncGrad::KPynativeOp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);

  auto &prim = grad_param->op_grad_info->op_prim;
  if (!PyNativeAlgo::AutoGrad::IsPrimNeedGrad(prim) ||
      (grad_by_value_ && !PyNativeAlgo::AutoGrad::NeedGrad(grad_param->op_grad_info->input_value))) {
    MS_LOG(DEBUG) << "Prim " << prim->name() << " does not need to do op grad.";
    return true;
  }
  auto flatten_inputs = PyNativeAlgo::DataConvert::FlattenTensorSeqInValueSeq(grad_param->op_grad_info->input_value);
  ConstructParameterNodes(flatten_inputs);
  BackwardNodePtr fn = nullptr;
  bool is_custom_prim =
    IsPrimitiveEquals(prim, prim::kPrimHookBackward) || IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook);
  if (!is_custom_prim) {
    auto handle = expander::bprop::BpropIRBuilderFactory::Instance().GetBuilder(prim->name());
    if (handle != nullptr) {
      fn = BuildFuncBackwardNode(prim, handle->func, flatten_inputs, grad_param->op_grad_info);
    } else {
      fn = BuildCustomBackwardNode(prim, flatten_inputs, grad_param->op_grad_info);
    }
  } else {
    PyNativeAlgo::AutoGrad::CheckRecomputeInputs(grad_param);
    fn = BuildHookBackwardNode(prim, flatten_inputs, grad_param->op_grad_info);
  }
  auto variable = std::make_shared<FuncVariable>(fn, false);
  if (isa<FakeBackwardNode>(fn)) {
    variable->set_is_fake_bprop(true);
    variable->set_fake_prim_name(prim->name());
  }

  (void)variable_set_.insert(variable);
  SetFlattenTensorGradMetaData(PyNativeAlgo::DataConvert::FlattenTensorSeqInValue(grad_param->op_grad_info->out_value),
                               variable);
  MS_LOG(DEBUG) << "End update next edge for " << variable->ToString();
  return true;
}

void FuncGrad::UpdateOutputNodeOfTopCell(const ValuePtr &sens_out) {
  MS_LOG(DEBUG) << "Real output of top cell is " << PyNativeAlgo::Common::GetIdByValue(sens_out);
  sens_value_ = sens_out;
  auto flatten_sens = PyNativeAlgo::DataConvert::FlattenTensorSeqInValue(sens_out);
  ConstructParameterNodes(flatten_sens);
}

void FuncGrad::BuildForwardLastNode(const ValuePtr &sens_gradient) {
  ValuePtrList root_gradient_value;
  if (sens_gradient == nullptr) {
    root_gradient_value = OnsLike(sens_value_);
  } else {
    root_gradient_value = PyNativeAlgo::DataConvert::FlattenTensorSeqInValue(sens_gradient);
  }
  auto root = std::make_shared<GraphRoot>("GraphRoot");
  auto flatten_args = PyNativeAlgo::DataConvert::FlattenTensorSeqInValue(sens_value_);
  root->UpdateNextEdges(flatten_args);
  root_gradients_ = root->BuildFlattenSensGradient(root_gradient_value);
  auto sens_variable = std::make_shared<FuncVariable>(root, false);
  if (root_gradients_.empty()) {
    sens_variable->set_is_need_grad(false);
  }
  (void)variable_set_.insert(sens_variable);
  last_variable_ = sens_variable;
}

bool FuncGrad::KPynativeWithFProp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  MS_LOG(DEBUG) << "Do KPynativeWithFProp";
  if (!grad_by_value_) {
    MS_LOG(EXCEPTION) << "High grad not support pyboost call";
  }
  auto fn = BuildGraphBackwardNode(grad_param);
  auto variable = std::make_shared<FuncVariable>(fn, false);
  (void)variable_set_.insert(variable);
  SetFlattenTensorGradMetaData(PyNativeAlgo::DataConvert::FlattenTensorSeqInValue(grad_param->op_grad_info->out_value),
                               variable);
  return true;
}

BackwardNodePtr FuncGrad::BuildGraphBackwardNode(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  if (ir_bprop_ == nullptr) {
    ir_bprop_ = std::make_unique<IrBprop>(std::make_shared<AdParam>(), device_target_, grad_by_value_);
  }
  grad_param->is_func_grad = true;
  auto [cache_hit, bprop_graph] = ir_bprop_->GetBpropGraph(grad_param);
  bool is_jit_dynamic_shape = grad_param->is_jit_graph && grad_param->use_dynamic_shape_process;
  // Save replace info in first time
  if (!cache_hit && is_jit_dynamic_shape && grad_param->has_added_v) {
    const auto &jit = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor()->jit();
    jit->SaveForwardOutputTensorInfoInBpropGraph(bprop_graph);
  }
  VectorRef input_args;
  (void)std::transform(grad_param->op_grad_info->input_value.begin(), grad_param->op_grad_info->input_value.end(),
                       std::back_inserter(input_args), [](const ValuePtr &v) { return v; });
  PyNativeAlgo::Common::DumpGraphIR("call_graph.ir", bprop_graph);
  auto fn = std::make_shared<GraphBackwardNode>(
    bprop_graph->ToString(), bprop_graph, input_args, grad_param->op_grad_info->out_value,
    grad_param->op_grad_info->output_size, grad_param->graph_cache_key, grad_param->is_control_flow,
    grad_param->is_jit_graph, grad_param->use_dynamic_shape_process, grad_param->jit_out_has_dict);
  auto flatten_inputs = PyNativeAlgo::DataConvert::FlattenTensorSeqInValueSeq(grad_param->op_grad_info->input_value);
  ConstructParameterNodes(flatten_inputs);
  fn->UpdateNextEdges(flatten_inputs);
  return fn;
}

void FuncGrad::BackPropagate() {
  MS_LOG(DEBUG) << "Begin BackPropagate";
  const auto &last_node_reverse_iter = GetLastNodeReverseIter();
  const auto &root_fn = (*last_node_reverse_iter)->func_node();
  mindspore::HashMap<BackwardNode *, ValuePtrList> input_buffer;
  (void)input_buffer.insert({root_fn.get(), root_gradients_});
  MS_LOG(DEBUG) << "Is running recompute grad " << is_run_recompute_;
  for (auto iter = last_node_reverse_iter; iter != variable_set_.rend(); ++iter) {
    const auto &variable = *iter;
    const auto &fn = variable->func_node();
    MS_LOG(DEBUG) << "Begin calculate op: " << fn->name() << " gradients!";
    if (!variable->is_need_propagate() || !variable->is_need_grad()) {
      MS_LOG(DEBUG) << "No need grad, variable is: " << variable->ToString();
      continue;
    }
    if (static_cast<bool>(MS_UNLIKELY(variable->is_fake_bprop()))) {
      MS_LOG(EXCEPTION) << "Illegal primitive " << variable->fake_prim_name() << "'s bprop not defined";
    }
    auto gradient_in_iter = input_buffer.find(fn.get());
    if (gradient_in_iter == input_buffer.end()) {
      MS_LOG(EXCEPTION) << "Fn not has gradient";
    }
    auto &gradient_in = gradient_in_iter->second;
    MS_LOG(DEBUG) << PyNativeAlgo::Common::PrintDebugInfo(gradient_in, "Begin print gradient in: ");
    // If register hook by weight, and weight in recompute cell.So, hook will execute, which is not expect.
    if (!is_run_recompute_) {
      gradient_in = CallBackwardHooks(fn->op_output(), &gradient_in);
    }
    auto gradient_out = fn->CallBackward(gradient_in);
    MS_LOG(DEBUG) << PyNativeAlgo::Common::PrintDebugInfo(gradient_out, "Begin print gradient out: ");
    if (gradient_out.size() != fn->next_edges().size()) {
      MS_LOG(EXCEPTION) << "Fn gradient size should be same as next edges size";
    }
    for (size_t i = 0; i < fn->next_edges().size(); ++i) {
      const auto &next_edge = fn->next_edges()[i];
      const auto &last_variable = next_edge.variable;
      // If network not calculate inputs grad, some op will be pruning, we need skip this op.
      if (!last_variable->is_need_grad()) {
        MS_LOG(DEBUG) << "variable is not need grad, " << last_variable->ToString();
        continue;
      }
      const auto &last_fn = last_variable->func_node();
      const auto &last_gradient = gradient_out[i];
      // If last_gradient is None, It represents that this tensor grad is zeros.
      if (last_gradient->isa<None>()) {
        MS_LOG(DEBUG) << last_variable->ToString() << ", its gradient is kNone!";
        continue;
      }
      if (input_buffer.find(last_fn.get()) != input_buffer.end()) {
        Add(last_gradient, next_edge.input_index, func_impl_, &input_buffer[last_fn.get()]);
      } else {
        input_buffer[last_fn.get()] =
          PaddingGradientInput(last_gradient, last_fn->output_size(), next_edge.input_index);
      }
      last_variable->set_is_need_propagate(true);
    }
    if (variable->is_leaf()) {
      const auto &grads = input_buffer[fn.get()];
      MS_LOG(DEBUG) << "Get leaf node " << variable->ToString();
      if (grads.empty() || grads[0]->isa<None>()) {
        MS_LOG(EXCEPTION) << variable->ToString() << ", " << (grads.empty() ? "grad is empty" : "grad is kNone");
      }
      auto grad_tensor = grads[0]->cast<tensor::BaseTensorPtr>();
      MS_EXCEPTION_IF_NULL(grad_tensor);
      variable->set_grad(grad_tensor);
    }
    (void)input_buffer.erase(fn.get());
    ReleaseResource(variable);
  }
  MS_LOG(DEBUG) << "End BackPropagate";
}

OrderedSet<FuncVariablePtr>::reverse_iterator FuncGrad::GetLastNodeReverseIter() {
  for (auto iter = variable_set_.rbegin(); iter != variable_set_.rend(); ++iter) {
    if (*iter == last_variable_) {
      last_variable_->set_is_need_propagate(true);
      return iter;
    }
  }
  return variable_set_.rend();
}

void FuncGrad::ConstructParameterNodes(const ValuePtrList &inputs) {
  for (const auto &value : inputs) {
    if (value->isa<tensor::BaseTensor>()) {
      const auto &tensor = value->cast<tensor::BaseTensorPtr>();
      const auto &auto_grad_meta_data = tensor->auto_grad_meta_data();
      // Get scalar tensor
      if (auto_grad_meta_data == nullptr || auto_grad_meta_data->variable() != nullptr) {
        continue;
      }
      if (PyNativeAlgo::Common::IsParam(auto_grad_meta_data->input_type())) {
        param_meta_grad_info_.emplace_back(tensor, auto_grad_meta_data);
      }
      if (auto_grad_meta_data->input_type() == InputType::kParameter &&
          PyNativeAlgo::Common::IsParamRequiresGrad(tensor)) {
        auto fn = std::make_shared<BackwardNode>("parameter");
        fn->set_op_output(value);
        auto variable = std::make_shared<FuncVariable>(fn, true);
        auto_grad_meta_data->set_variable(variable);
        (void)variable_set_.insert(variable);
        weights_used_in_graph_.emplace_back(tensor);
      }
    }
  }
}

BackwardNodePtr FuncGrad::BuildFuncBackwardNode(const PrimitivePtr &prim, const expander::bprop::BpropBuilderFunc &func,
                                                const ValuePtrList &flatten_inputs, const OpGradInfoPtr &op_grad_info) {
  PyNativeAlgo::AutoGrad::CheckAndSetAbstract(op_grad_info);
  auto emitter = std::make_shared<FuncBuilder>(prim->name(), device_target_, nullptr);
  auto node_inputs = GenerateNodeInputs(op_grad_info, emitter);
  auto fn = std::make_shared<FuncBackwardNode>(prim->name(), func, emitter, prim->attrs(), node_inputs,
                                               op_grad_info->output_size);
  fn->UpdateNextEdges(flatten_inputs);
  return fn;
}

BackwardNodePtr FuncGrad::BuildCustomBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                                  const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(DEBUG) << "Try build custom bprop: " << prim->name();
  {
    py::gil_scoped_acquire gil;
    auto prim_py = prim->cast<PrimitivePyPtr>();
    if (prim_py == nullptr) {
      MS_LOG(DEBUG) << "Prim is not PrimitivePy, can not find python bprop";
      return BuildFakeBackwardNode(prim, flatten_inputs, op_grad_info);
    }
    py::function fn = prim_py->GetBpropFunction();
    if (py::isinstance<py::none>(fn)) {
      fn = GetBpropFunction(prim->name());
    }
    if (!fn || py::isinstance<py::none>(fn)) {
      MS_LOG(INFO) << "Can not find bprop function for " << prim->name() << ". fn: " << ConvertPyObjToString(fn);
      return BuildFakeBackwardNode(prim, flatten_inputs, op_grad_info);
    }
    (void)prim_py->AddBackwardHookFn(0, fn);
    (void)prim_py->AddAttr("custom_op_bprop", MakeValue(true));
  }
  return BuildHookBackwardNode(prim, flatten_inputs, op_grad_info);
}

BackwardNodePtr FuncGrad::BuildHookBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                                const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(prim);
  auto bprop_cut = PyNativeAlgo::AutoGrad::BuildBpropCutPrim(prim, op_grad_info->is_need_recompute);
  VectorRef args = GeneratePythonArgs(op_grad_info);
  auto fn = std::make_shared<HookBackwardNode>(prim->name(), bprop_cut, std::move(args), op_grad_info->output_size,
                                               op_grad_info->out_abs);
  fn->UpdateNextEdges(flatten_inputs);
  return fn;
}

BackwardNodePtr FuncGrad::BuildFakeBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                                const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(prim);
  auto fn = std::make_shared<FakeBackwardNode>(prim->name(), op_grad_info->output_size);
  fn->UpdateNextEdges(flatten_inputs);
  return fn;
}

ValuePtr FuncGrad::GetGrads(const tensor::BaseTensorPtrList &weights, const std::vector<size_t> &grad_position,
                            const GradAttr &grad_attr) {
  auto inputs_grad = GetInputGrads(grad_attr.grad_all_inputs, grad_attr.get_by_position, grad_position);
  auto weights_grad = GetWeightGrads(grad_attr.grad_weights, weights, grad_attr.weight_param_is_tuple);
  // Gradients wrt inputs and weights.
  if (inputs_grad != nullptr && weights_grad != nullptr) {
    if (IsOutputBothEmpty(inputs_grad, weights_grad)) {
      return GenerateEmptyTupleValue();
    }
    ValuePtrList gradients{inputs_grad, weights_grad};
    return std::make_shared<ValueTuple>(gradients);
  }
  // Gradients wrt inputs.
  if (inputs_grad != nullptr) {
    return inputs_grad;
  }
  // Gradients wrt weights.
  if (weights_grad != nullptr) {
    return weights_grad;
  }
  // grad_all_inputs, grad_weights and get_by_position are all false.
  if (cell_inputs_.empty()) {
    // If no input nodes, return empty tuple.
    return std::make_shared<ValueTuple>(ValuePtrList{});
  }

  // If there are input nodes, return gradient of first input node.
  // Tuple, List, scalar will be ignore
  if (IsValidTensorInput(cell_inputs_[kIndex0].first)) {
    return PyNativeAlgo::AutoGrad::BuildSpecialValueGrad(
      cell_inputs_[kIndex0].first, cell_inputs_[kIndex0].second->grad(), func_impl_.get(), SpecialType::kZerosLikeType);
  }
  MS_LOG(DEBUG) << "Get first input node is not tensor " << cell_inputs_[0].first->ToString();
  return std::make_shared<ValueTuple>(ValuePtrList{});
}

ValuePtr FuncGrad::GetInputGrads(bool grad_all_inputs, bool get_by_position, const std::vector<size_t> &grad_position) {
  std::vector<size_t> grad_pos_list;
  if (get_by_position) {
    grad_pos_list = grad_position;
  } else if (grad_all_inputs) {
    grad_pos_list.resize(cell_inputs_.size());
    iota(grad_pos_list.begin(), grad_pos_list.end(), 0);
  } else {
    return nullptr;
  }
  ValuePtrList input_grads;
  input_grads.reserve(cell_inputs_.size());
  if (!cell_inputs_.empty()) {
    for (size_t index : grad_pos_list) {
      if (index >= cell_inputs_.size()) {
        MS_LOG(EXCEPTION) << "Position index " << index << " is exceed input size.";
      }
      // Tuple, List, scalar will be ignore
      if (!IsValidTensorInput(cell_inputs_[index].first)) {
        MS_LOG(DEBUG) << cell_inputs_[index].first->ToString() << "is no tensor";
        continue;
      }
      ValuePtr real_dout = PyNativeAlgo::AutoGrad::BuildSpecialValueGrad(
        cell_inputs_[index].first, cell_inputs_[index].second->grad(), func_impl_.get(), SpecialType::kZerosLikeType);
      (void)input_grads.emplace_back(real_dout);
    }
    if (get_by_position && input_grads.size() == kSizeOne) {
      return input_grads[kIndex0];
    }
  }
  return std::make_shared<ValueTuple>(input_grads);
}

ValuePtr FuncGrad::GetWeightGrads(bool grad_weights, const tensor::BaseTensorPtrList &weights,
                                  bool weight_param_is_tuple) {
  // No need to return gradient of weights.
  if (!grad_weights) {
    return nullptr;
  }
  if (weight_param_is_tuple) {
    ValuePtrList weight_grads;
    weight_grads.reserve(weights.size());
    for (const auto &weight : weights) {
      (void)weight_grads.emplace_back(GetWeightGrad(weight));
    }
    return std::make_shared<ValueTuple>(weight_grads);
  }
  return GetWeightGrad(weights[0]);
}

ValuePtr FuncGrad::GetWeightGrad(const tensor::BaseTensorPtr &weight) {
  MS_EXCEPTION_IF_NULL(weight);
  auto auto_grad_meta_data = weight->auto_grad_meta_data();
  if (auto_grad_meta_data == nullptr) {
    return func_impl_->Zeros(weight);
  }
  auto variable = auto_grad_meta_data->variable();
  const auto &func_variable = std::dynamic_pointer_cast<FuncVariable>(variable);
  MS_LOG(DEBUG) << "Get variable " << (variable != nullptr ? variable->ToString() : "is nullptr");
  if (variable != nullptr && variable->is_need_grad()) {
    // If weight used in the forward network, but requires_grad is false, return zero like.
    if (func_variable->grad() == nullptr ||
        (weight->param_info() != nullptr && !weight->param_info()->requires_grad())) {
      MS_LOG(INFO) << "weight participate in forward calculation, but requires_grad is false";
      return func_impl_->Zeros(weight);
    }
    auto weight_grad = func_variable->grad();
    return weight_grad;
  }
  MS_LOG(INFO) << "weight not participate in forward calculation, but requires grad, id: "
               << PyNativeAlgo::Common::GetIdByValue(weight);
  return func_impl_->Zeros(weight);
}

void FuncGrad::ClearGrads(const tensor::BaseTensorPtrList &weights) {
  // Clear input grads.
  for (const auto &input : cell_inputs_) {
    input.second->set_grad(nullptr);
  }
  cell_inputs_.clear();
}

ValuePtrList FuncGrad::OnsLike(const ValuePtr &sens) {
  MS_EXCEPTION_IF_NULL(sens);
  auto flatten_values = PyNativeAlgo::DataConvert::FlattenTensorSeqInValue(sens);
  const auto &v = PyNativeAlgo::AutoGrad::BuildSpecialValueGrad(std::make_shared<ValueTuple>(flatten_values), nullptr,
                                                                func_impl_.get(), SpecialType::kOnesLikeType);
  auto v_seq = v->cast<ValueTuplePtr>();
  return v_seq->value();
}

void FuncGrad::CheckSensShapeAndType(const ValuePtr &sens_gradient) {
  if (sens_gradient == nullptr) {
    return;
  }
  const auto sens_gradient_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(sens_gradient->ToAbstract());
  const auto out_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(sens_value_->ToAbstract());
  const auto &sens_gradient_shape = sens_gradient_abs->BuildShape()->ToString();
  const auto &out_shape = out_abs->BuildShape()->ToString();
  if (sens_gradient_shape != "()" && out_shape != "()") {
    if (sens_gradient_shape != out_shape) {
      // Sens shape in ir graph is determined by graph output, so it can be dynamic shape; But input shape is
      // determined by user input, which could not be dynamic shape.
      if (!sens_gradient_abs->BuildShape()->IsDynamic()) {
        MS_EXCEPTION(ValueError) << "The shape should be " << out_shape << ", but got " << sens_gradient_shape << ", "
                                 << ", sens gradient abs " << sens_gradient_abs->ToString() << ", out abs"
                                 << out_abs->ToString();
      }
    }
    const auto &sens_gradient_dtype = sens_gradient_abs->BuildType()->ToString();
    const auto &out_dtype = out_abs->BuildType()->ToString();
    if (sens_gradient_dtype != out_dtype) {
      MS_EXCEPTION(TypeError) << "The dtype should be " << out_dtype << ", but got " << sens_gradient_dtype << ", "
                              << ", sens gradient abs " << sens_gradient_abs->ToString() << ", out abs"
                              << out_abs->ToString();
    }
  }
}

void FuncGrad::PruningGradGraph(const tensor::BaseTensorPtrList &weights, const GradAttr &grad_attr,
                                const std::vector<size_t> &grad_position) {
  PruningInput(grad_attr, grad_position);
  PruningWeights(weights, grad_attr);

  // Pruning all node in grad graph
  for (const auto &variable : variable_set_) {
    if (variable->is_leaf()) {
      continue;
    }
    bool is_need_grad =
      std::any_of(variable->func_node()->next_edges().begin(), variable->func_node()->next_edges().end(),
                  [](const auto &edge) { return edge.variable->is_need_grad(); });
    if (!is_need_grad) {
      variable->set_is_need_grad(false);
    }
  }
}

void FuncGrad::PruningInput(const GradAttr &grad_attr, const std::vector<size_t> &grad_position) {
  mindspore::HashSet<size_t> grad_pos_list{grad_position.begin(), grad_position.end()};
  // Pruning inputs by position in grad graph
  if (grad_attr.get_by_position) {
    for (size_t i = 0; i < cell_inputs_.size(); ++i) {
      if (grad_pos_list.find(i) == grad_pos_list.end()) {
        cell_inputs_[i].second->set_is_need_grad(false);
      }
    }
    return;
  }

  // Pruning first input in grad graph
  if (!grad_attr.grad_all_inputs && !grad_attr.get_by_position && !grad_attr.grad_weights) {
    for (size_t i = 1; i < cell_inputs_.size(); ++i) {
      cell_inputs_[i].second->set_is_need_grad(false);
    }
  }

  // Pruning all inputs not grad
  if (!grad_attr.grad_all_inputs && grad_attr.grad_weights) {
    for (auto &cell_input : cell_inputs_) {
      cell_input.second->set_is_need_grad(false);
    }
  }
}

void FuncGrad::PruningWeights(const tensor::BaseTensorPtrList &weights, const GradAttr &grad_attr) {
  // Pruning weights in grad graph
  if (grad_attr.grad_weights) {
    mindspore::HashSet<std::string> grad_weights_id;
    for (const auto &weight : weights) {
      (void)grad_weights_id.emplace(weight->id());
    }
    for (const auto &weight : weights_used_in_graph_) {
      if (grad_weights_id.find(weight->id()) == grad_weights_id.end()) {
        auto variable = weight->auto_grad_meta_data()->variable();
        MS_EXCEPTION_IF_NULL(variable);
        variable->set_is_need_grad(false);
      }
    }
  } else {
    for (const auto &weight : weights_used_in_graph_) {
      auto variable = weight->auto_grad_meta_data()->variable();
      MS_EXCEPTION_IF_NULL(variable);
      variable->set_is_need_grad(false);
    }
  }
}

ValuePtr FuncGrad::Finish(const tensor::BaseTensorPtrList &weights, const std::vector<size_t> &grad_position,
                          const GradAttr &grad_attr, const ValuePtr &sens) {
  CheckSensShapeAndType(sens);
  BuildForwardLastNode(sens);
  PruningGradGraph(weights, grad_attr, grad_position);
  if (last_variable_->is_need_grad()) {
    GilReleaseWithCheck gil_release;
    BackPropagate();
  }
  PyNativeAlgo::Common::DumpGraphIR("func_grad.ir", std::make_shared<FuncGraph>());
  python_adapter::PyAdapterCallback::ProcessUnPairedCellHook(true);
  ValuePtr gradients = GetGrads(weights, grad_position, grad_attr);
  ClearGrads(weights);
  return gradients;
}
}  // namespace mindspore::pynative::autograd
