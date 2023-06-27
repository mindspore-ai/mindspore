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

#include "pipeline/pynative/forward/forward.h"
#include <set>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/python_fallback_running.h"
#include "backend/graph_compiler/transform.h"
#include "utils/ms_context.h"
#include "mindrt/include/fork_utils.h"
#include "pipeline/pynative/forward/forward_task.h"
#include "pipeline/pynative/predict_out_type_map.h"
#include "include/common/utils/stub_tensor.h"
#include "runtime/pynative/op_executor.h"
#ifndef ENABLE_SECURITY
#include "include/backend/debug/profiler/profiling.h"
using mindspore::profiler::ProfilerManager;
#endif
#include "include/common/utils/tensor_future.h"

namespace mindspore {
namespace pynative {
namespace {
const std::set<std::string> kVmOperators = {"InsertGradientOf", "StopGradient", "HookBackward", "CellBackwardHook"};
constexpr char kBegin[] = "Begin";
constexpr char kEnd[] = "End";
enum class RunOpArgsEnum : size_t { PY_PRIM = 0, PY_NAME, PY_INPUTS, PY_ARGS_NUM };

// Shallow Copy Value and change shape
ValuePtr ShallowCopyValue(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value);
  auto tensor_abs = op_run_info->base_op_run_info.abstract;
  if (tensor_abs->isa<abstract::AbstractRefTensor>()) {
    tensor_abs = tensor_abs->cast<abstract::AbstractRefPtr>()->CloneAsTensor();
  }
  auto new_shape = tensor_abs->BuildShape()->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(new_shape);
  if (value->isa<mindspore::tensor::Tensor>()) {
    auto tensor_value = value->cast<mindspore::tensor::TensorPtr>();
    return std::make_shared<mindspore::tensor::Tensor>(tensor_value->data_type(), new_shape->shape(),
                                                       tensor_value->data_c(), tensor_value->Size());
  } else if (value->isa<ValueTuple>()) {
    std::vector<ValuePtr> values;
    auto value_tuple = value->cast<ValueTuplePtr>();
    (void)std::transform(value_tuple->value().begin(), value_tuple->value().end(), std::back_inserter(values),
                         [op_run_info](const ValuePtr &elem) { return ShallowCopyValue(op_run_info, elem); });
    return std::make_shared<ValueTuple>(values);
  } else {
    return value;
  }
}

/// TODO(caifubi): delete and throw exception in Init().
ValuePtr CopyTensorValueWithNewId(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    // This constructor will make a tensor with the new id
    auto new_tensor = std::make_shared<tensor::Tensor>(tensor->data_type(), tensor->shape(), tensor->data_ptr());
    new_tensor->set_device_address(tensor->device_address());
    new_tensor->set_sync_status(tensor->sync_status());
    new_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
    return new_tensor;
  } else if (v->isa<ValueSequence>()) {
    const auto &v_seq = v->cast<ValueSequencePtr>();
    ValuePtrList v_list;
    for (const auto &ele : v_seq->value()) {
      (void)v_list.emplace_back(CopyTensorValueWithNewId(ele));
    }
    return std::make_shared<ValueTuple>(v_list);
  } else {
    return v;
  }
}

MsBackendPolicy GetBackendPolicy(const std::string &device_target) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MsBackendPolicy backend_policy = kMsBackendVmOnly;
  if (device_target == kAscendDevice) {
    if (ms_context->backend_policy() == "ge") {
      MS_LOG(EXCEPTION) << "In PyNative mode, not support ge backend!";
    }
#ifdef WITH_BACKEND
    const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());

    if (!device_context->GetDeprecatedInterface()->IsTsdOpened(ms_context)) {
      if (!device_context->GetDeprecatedInterface()->OpenTsd(ms_context)) {
        MS_LOG(EXCEPTION) << "Open tsd failed";
      }
    }
#endif
  }
  return backend_policy;
}

void UpdateOutputStubNodeAbs(const FrontendOpRunInfoPtr &op_run_info) {
  if (op_run_info->stub_output == nullptr) {
    return;
  }
  const auto &abs = op_run_info->base_op_run_info.abstract;
  auto success = op_run_info->stub_output->SetAbstract(abs);
  if (!success) {
    const auto &op_name = op_run_info->base_op_run_info.op_name;
    MS_EXCEPTION(TypeError) << "The predict type and infer type is not match, predict type is "
                            << PredictOutTypeByName(op_name) << ", infer type is " << abs->BuildType()
                            << ", the name of operator is [" << op_name
                            << "]. Please modify or add predict type of operator in predict_out_type_map.h.";
  }
  MS_LOG(DEBUG) << "Update StubNode abstract " << abs->ToString();
}

void ClonePrim(const FrontendOpRunInfoPtr &op_run_info) {
  // Clone a new prim
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto prim_py = op_run_info->op_grad_info->op_prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto new_adapter = std::make_shared<PrimitivePyAdapter>(*prim_py->adapter());
  auto new_prim = std::make_shared<PrimitivePy>(*(op_run_info->op_grad_info->op_prim->cast<PrimitivePyPtr>()));
  new_prim->EnableSharedMutex();
  op_run_info->op_grad_info->op_prim = new_prim;
  MS_EXCEPTION_IF_NULL(new_adapter);
  new_adapter->set_attached_primitive(new_prim);
}

bool IsDynamicInputs(const FrontendOpRunInfoPtr &op_run_info) {
  for (const auto &value : op_run_info->op_grad_info->input_value) {
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<stub::SequenceNode>()) {
      return true;
    }
    if (!value->isa<ValueSequence>()) {
      continue;
    }
    auto value_seq = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_seq);

    const auto &tuple_inputs = value_seq->value();
    if (tuple_inputs.empty()) {
      continue;
    }
    if (tuple_inputs[0]->isa<tensor::Tensor>() || tuple_inputs[0]->isa<stub::TensorNode>()) {
      return true;
    }
  }
  return false;
}

ValuePtr ConstructOutputInVM(const FrontendOpRunInfoPtr &op_run_info, const std::vector<ValuePtr> &result) {
  if (result.size() == 1 && op_run_info->base_op_run_info.abstract != nullptr &&
      !op_run_info->base_op_run_info.abstract->isa<abstract::AbstractSequence>()) {
    return result[kIndex0];
  }

  return std::make_shared<ValueTuple>(result);
}

void UpdateOutputStubNodeValue(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &out_value) {
  if (op_run_info->stub_output != nullptr) {
    op_run_info->stub_output->SetValue(out_value);
  }
}

BackendOpRunInfoPtr CreateBackendOpRunInfo(const FrontendOpRunInfoPtr &op_run_info) {
  auto backend_op_run_info =
    std::make_shared<BackendOpRunInfo>(op_run_info->base_op_run_info, op_run_info->op_grad_info->op_prim, true, false);
  backend_op_run_info->output_tensors = op_run_info->output_tensors;
  // Need to update promise in backend task.
  backend_op_run_info->device_sync_promises = std::move(op_run_info->device_sync_promises);
  return backend_op_run_info;
}

void TransformOutputValues(const FrontendOpRunInfoPtr &op_run_info) {
  std::vector<ValuePtr> output_values;
  for (auto &output_tensor : op_run_info->output_tensors) {
    if (op_run_info->requires_grad) {
      output_tensor->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>());
      output_tensor->auto_grad_meta_data()->set_grad_type(TensorGradType::kOpOutput);
    }
    output_values.emplace_back(output_tensor);
  }
  auto result_value = std::make_shared<ValueTuple>(output_values);
  if (result_value->size() == 1 && op_run_info->base_op_run_info.abstract != nullptr &&
      !op_run_info->base_op_run_info.abstract->isa<abstract::AbstractSequence>()) {
    op_run_info->real_out = result_value->value().front();
  } else {
    op_run_info->real_out = result_value;
  }
}

void CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs,
                        std::vector<DeviceAddressPromisePtr> *device_sync_promises) {
  auto create_tensor = [&outputs, &device_sync_promises](const TypePtr &type, const ShapeVector &shape_vector) {
    auto output_tensor = std::make_shared<tensor::Tensor>(type->type_id(), shape_vector);
    output_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
    outputs->emplace_back(output_tensor);
    MS_LOG(DEBUG) << "Create output tensor " << output_tensor->ToString();

    DeviceAddressPromisePtr promise =
      std::make_unique<DeviceAddressPromise>(std::promise<DeviceAddressFutureDataPtr>());
    auto future = promise->GetFuture();
    auto device_address_future = std::make_shared<DeviceAddressFuture>(std::move(future));
    output_tensor->set_address_future(device_address_future);
    (void)device_sync_promises->emplace_back(std::move(promise));
  };

  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<abstract::AbstractSequence>()) {
    auto seq = abstract->cast<abstract::AbstractSequencePtr>();
    auto elements = seq->elements();
    for (const auto &element : elements) {
      CreateOutputTensor(element, outputs, device_sync_promises);
    }
  } else if (abstract->isa<abstract::AbstractTensor>()) {
    auto abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
    auto shape = abstract_tensor->BuildShape();
    auto type = abstract_tensor->element()->BuildType();
    MS_LOG(DEBUG) << "get abstract tensor shape " << shape->ToString() << " type " << type->ToString();
    if (!shape->isa<abstract::Shape>()) {
      MS_LOG(EXCEPTION) << "AbstractTensor shape is valid " << shape->ToString();
    }
    auto shape_vector = shape->cast<abstract::ShapePtr>()->shape();
    create_tensor(type, shape_vector);
  } else if (abstract->isa<abstract::AbstractScalar>()) {
    auto scalar = abstract->cast<abstract::AbstractScalarPtr>();
    const auto &type = scalar->BuildType();
    MS_LOG(DEBUG) << "Create scalar tensor type " << type->ToString();
    create_tensor(type, {});
  } else {
    MS_LOG(EXCEPTION) << "Not support abstract " << abstract->ToString();
  }
}

void UpdateStubTensor(const FrontendOpRunInfoPtr &op_run_info) {
  // Some operators do not have StubNodes, such as Cast inserted for automatic mixed precision.
  if (op_run_info->stub_output != nullptr) {
    if (op_run_info->base_op_run_info.has_dynamic_output) {
      UpdateOutputStubNodeAbs(op_run_info);
    }
    op_run_info->stub_output->SetValue(op_run_info->real_out);
  }
}

bool EnableBackendAsync(const FrontendOpRunInfoPtr &op_run_info) {
  static const std::set<std::string> kInvalidInferResultOp = {kDropoutOpName};
  return kInvalidInferResultOp.find(op_run_info->base_op_run_info.op_name) == kInvalidInferResultOp.end() &&
         !op_run_info->base_op_run_info.has_dynamic_output;
}
}  // namespace

void ForwardExecutor::ClearForwardTask() {
  if (frontend_queue_ != nullptr) {
    GilReleaseWithCheck gil_release;
    frontend_queue_->Clear();
  }
  if (backend_queue_ != nullptr) {
    GilReleaseWithCheck gil_release;
    backend_queue_->Clear();
  }
}

void ForwardExecutor::WaitForwardTask() {
  if (frontend_queue_ != nullptr) {
    GilReleaseWithCheck gil_release;
    frontend_queue_->Wait();
  }
  if (backend_queue_ != nullptr) {
    GilReleaseWithCheck gil_release;
    backend_queue_->Wait();
  }
}

bool ForwardExecutor::IsVmOp(const std::string &op_name) const {
  return kVmOperators.find(op_name) != kVmOperators.end();
}

std::string ForwardExecutor::GetCurrentCellObjId() const {
  if (forward_cell_stack_.empty()) {
    return "";
  }
  auto &cell = forward_cell_stack_.top();
  return cell->id();
}

GradExecutorPtr ForwardExecutor::grad() const {
  auto grad_executor = grad_executor_.lock();
  MS_EXCEPTION_IF_NULL(grad_executor);
  return grad_executor;
}

void ForwardExecutor::ReInit() {
  device_target_ = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  enable_async_ = !MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
}

void ForwardExecutor::Init() {
  // Single op run with out cell or function packed
  if (MS_UNLIKELY(infer_operation()->only_single_op_run())) {
    ReInit();
  }
  if (init_) {
    return;
  }
  init_ = true;
  MS_LOG(DEBUG) << "Init ForwardExecutor";
  compile::SetMindRTEnable();
  python_adapter::set_python_env_flag(true);
  runtime::OpExecutor::GetInstance().RegisterForwardCallback([this]() {
    frontend_queue_->Wait();
    backend_queue_->Wait();
  });
  // If the fork occurs, device resources need to be released in child process.
  ForkUtils::GetInstance().RegisterCallbacks(this, static_cast<void (ForwardExecutor::*)()>(nullptr),
                                             static_cast<void (ForwardExecutor::*)()>(nullptr),
                                             &ForwardExecutor::ReinitAfterFork);
}

bool ForwardExecutor::EnablePipeline(const std::string &op_name) const {
#if defined(ENABLE_TEST) || defined(__APPLE__)
  return false;
#else
  return !IsVmOp(op_name) && op_name != "Custom" && !ScopedFallbackRunning::on() && enable_async() &&
         MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode;
#endif
}

void ForwardExecutor::DispatchFrontendTask(const FrontendOpRunInfoPtr &op_run_info) {
  auto forward_task = std::make_shared<FrontendTask>(
    [this](const FrontendOpRunInfoPtr &op_run_info) { RunOpFrontend(op_run_info); }, op_run_info);
  frontend_queue_->Push(forward_task);
}

void ForwardExecutor::DispatchBackendTask(const FrontendOpRunInfoPtr &op_run_info,
                                          const session::BackendOpRunInfoPtr &backend_op_run_info) {
  static auto run_backend_with_grad = [this](const FrontendOpRunInfoPtr &op_run_info,
                                             const session::BackendOpRunInfoPtr &backend_op_run_info) {
    // Update tensor device address in backend.
    RunOpBackendInner(op_run_info, backend_op_run_info);

    if (!op_run_info->requires_grad) {
      MS_LOG(DEBUG) << "Grad flag is false";
      return;
    }
    // 4. Do op grad and record op info
    // If ms function is compile, op info will not be find in second training step
    if (!op_run_info->async_status.is_ms_function_compiling && op_run_info->async_status.custom_bprop_cell_count <= 0) {
      grad()->ProcessOpGradInfo(op_run_info);
    }
  };

  auto backend_task = std::make_shared<BackendTask>(run_backend_with_grad, op_run_info, backend_op_run_info);
  backend_queue_->Push(backend_task);
}

void ForwardExecutor::RunOpFrontend(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "RunOp name: " << op_run_info->base_op_run_info.op_name;
  // Convert StubNode to Tensor and no need to concern about input StubNode anymore in this thread.
  PyNativeAlgo::Common::StubNodeToValue(op_run_info);
  // 1.Set cast for inputs
  SetCastForInputs(op_run_info);
  // 2.Infer output abstract
  InferOutputAbstract(op_run_info);

  if (!op_run_info->base_op_run_info.has_dynamic_output) {
    // Output is dynamic shape, need to SetAbstract after RunOp.
    UpdateOutputStubNodeAbs(op_run_info);
  }

  if (op_run_info->output_get_by_infer_value) {
    UpdateOutputStubNodeValue(op_run_info, op_run_info->real_out);
    MS_LOG(DEBUG) << "Grad flag: " << op_run_info->requires_grad
                  << " output_get_by_infer_value: " << op_run_info->output_get_by_infer_value;
    return;
  }

  PrepareOpInputs(op_run_info);
  if (EnableBackendAsync(op_run_info) && EnablePipeline(op_run_info->base_op_run_info.op_name)) {
    PrepareOpOutputs(op_run_info);
    const auto &backend_op_run_info = CreateBackendOpRunInfo(op_run_info);
    DispatchBackendTask(op_run_info, backend_op_run_info);
  } else {
    RunOpBackendSync(op_run_info);
  }
}

void ForwardExecutor::RunOpBackendSync(const FrontendOpRunInfoPtr &op_run_info) {
  backend_queue_->Wait();
  const auto &backend_op_run_info = CreateBackendOpRunInfo(op_run_info);
  RunOpBackend(op_run_info, backend_op_run_info);
  if (!op_run_info->requires_grad) {
    MS_LOG(DEBUG) << "Grad flag is false";
    UpdateStubTensor(op_run_info);
    return;
  }
  // 4. Do op grad and record op info
  // If ms function is compile, op info will not be find in second training step
  if (!op_run_info->async_status.is_ms_function_compiling && op_run_info->async_status.custom_bprop_cell_count <= 0) {
    grad()->ProcessOpGradInfo(op_run_info);
  }
  // output is dynamic shape. Need to update abstract and value.
  UpdateStubTensor(op_run_info);
}

FrontendOpRunInfoPtr ForwardExecutor::GenerateOpRunInfo(const py::args &args, bool stub) {
  if (args.size() != static_cast<size_t>(RunOpArgsEnum::PY_ARGS_NUM)) {
    MS_LOG(EXCEPTION) << "Three args are needed by RunOp";
  }
  Init();
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  // Used for async run
  op_run_info->base_op_run_info.op_name = args[static_cast<size_t>(RunOpArgsEnum::PY_NAME)].cast<std::string>();
  op_run_info->requires_grad = grad()->RequiresGrad();
  if (op_run_info->requires_grad) {
    op_run_info->base_op_run_info.use_dynamic_shape_process = grad()->use_dynamic_shape_process();
  } else {
    op_run_info->base_op_run_info.use_dynamic_shape_process = grad()->forward_use_dynamic_shape_process();
  }
  op_run_info->base_op_run_info.lazy_build = lazy_build_;
  PyNativeAlgo::PyParser::SetPrim(op_run_info, args[static_cast<size_t>(RunOpArgsEnum::PY_PRIM)]);
  PyNativeAlgo::PyParser::ParseOpInputByPythonObj(op_run_info, args[static_cast<size_t>(RunOpArgsEnum::PY_INPUTS)],
                                                  stub);
  op_run_info->base_op_run_info.device_target = GetCurrentDeviceTarget(op_run_info->op_grad_info->op_prim);
  bool is_dynamic_shape =
    op_run_info->base_op_run_info.has_dynamic_output || op_run_info->base_op_run_info.use_dynamic_shape_process;
  PyNativeAlgo::Common::GetConstInputToAttr(op_run_info->op_grad_info->op_prim, op_run_info->base_op_run_info.op_name,
                                            op_run_info->base_op_run_info.device_target, is_dynamic_shape,
                                            &op_run_info->input_to_attr);
  bool is_dynamic_inputs = IsDynamicInputs(op_run_info);
  if (!op_run_info->input_to_attr.empty() || is_dynamic_inputs) {
    MS_LOG(DEBUG) << "Op_prim need clone:" << op_run_info->base_op_run_info.op_name
                  << ", is_dynamic_inputs:" << is_dynamic_inputs
                  << ", input_to_attr is not empty:" << (!op_run_info->input_to_attr.empty());
    ClonePrim(op_run_info);
  }
  op_run_info->cell_obj_id = GetCurrentCellObjId();
  return op_run_info;
}

void ForwardExecutor::SetCastForInputs(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  // No need cast self
  if (op_run_info->base_op_run_info.op_name == prim::kPrimCast->name()) {
    return;
  }
  cast_operation()->DoCast(op_run_info);
}

void ForwardExecutor::ClearNodeAbsMap() const { infer_operation()->ClearNodeAbsCache(); }

void ForwardExecutor::SetNodeAbsMapByValue(const FrontendOpRunInfoPtr &op_run_info) const {
  infer_operation()->SetNodeAbsCacheByValue(op_run_info);
}

void ForwardExecutor::SetNodeAbsMapById(const std::string &id, const abstract::AbstractBasePtr &abs) const {
  infer_operation()->SetNodeAbsCacheById(id, abs);
}

AbstractBasePtr ForwardExecutor::GetNodeAbsById(const std::string &id) const {
  return infer_operation()->GetNodeAbsById(id);
}

void ForwardExecutor::InferOutputAbstract(const FrontendOpRunInfoPtr &op_run_info) const {
  infer_operation()->DoInfer(op_run_info);
}

VectorRef ForwardExecutor::RunOpBackendInner(const FrontendOpRunInfoPtr &op_run_info,
                                             const BackendOpRunInfoPtr &backend_op_run_info) {
  MS_LOG(DEBUG) << "RunOpBackendInner start";
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_id_ = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, true);
  // get graph info for checking it whether existing in the cache
  backend_op_run_info->base_op_run_info.graph_info = pynative::OpCompiler::GetInstance().GetSingleOpGraphInfo(
    backend_op_run_info->base_op_run_info, backend_op_run_info->op_prim);

#if defined(__APPLE__)
  backend_op_run_info->base_op_run_info.lazy_build = false;
#endif

  VectorRef outputs;
  const auto &cur_mind_rt_backend = GetMindRtBackend(op_run_info->base_op_run_info.device_target);
  MS_EXCEPTION_IF_NULL(cur_mind_rt_backend);
  bool use_dynamic_shape_process = op_run_info->base_op_run_info.use_dynamic_shape_process;
  if (use_dynamic_shape_process) {
    cur_mind_rt_backend->RunOpDynamic(backend_op_run_info, &outputs);
  } else {
    cur_mind_rt_backend->RunOp(backend_op_run_info, &outputs);
  }

  if (op_run_info->base_op_run_info.has_dynamic_output) {
    op_run_info->base_op_run_info.abstract = backend_op_run_info->base_op_run_info.abstract;
  }
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  MS_LOG(DEBUG) << "RunOpBackendInner end";
  return outputs;
}

void ForwardExecutor::RunOpBackend(const FrontendOpRunInfoPtr &op_run_info,
                                   const BackendOpRunInfoPtr &backend_op_run_info) {
  // Run op with selected backend, nop is no need run backend
  op_run_info->real_out = RunOpWithBackendPolicy(op_run_info, backend_op_run_info);
  // Not use GetNext abs
  if (op_run_info->base_op_run_info.op_name != kGetNextOpName) {
    op_run_info->out_value_id = PyNativeAlgo::Common::GetIdByValue(op_run_info->real_out);
    SetNodeAbsMapByValue(op_run_info);
  }
}

compile::MindRTBackendPtr ForwardExecutor::GetMindRtBackend(const string &cur_device_target) {
  const auto iter = mindrt_backends_.find(cur_device_target);
  if (iter != mindrt_backends_.end()) {
    return iter->second;
  } else {
    std::lock_guard<std::mutex> guard(pipeline::Resource::GetBackendInitMutex());
    auto backend = std::make_shared<compile::MindRTBackend>("ms", cur_device_target, device_id_);
    MS_EXCEPTION_IF_NULL(backend);
    mindrt_backends_[cur_device_target] = backend;
    return backend;
  }
}

ValuePtr ForwardExecutor::RunOpWithBackendPolicy(const FrontendOpRunInfoPtr &op_run_info,
                                                 const BackendOpRunInfoPtr &backend_op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  ValuePtr result;
  auto backend_policy = GetBackendPolicy(device_target_);
  if (backend_policy == kMsBackendVmOnly) {
#ifndef ENABLE_TEST
    if (IsVmOp(op_run_info->base_op_run_info.op_name)) {
      result = RunOpInVM(op_run_info);
    } else {
      result = RunOpInMs(op_run_info, backend_op_run_info);
    }
#else
    result = RunOpInVM(op_run_info);
#endif
  }
  return result;
}

ValuePtr ForwardExecutor::RunOpInVM(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_LOG(DEBUG) << "RunOpInVM start";
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->run_in_vm = true;
  if (op_run_info->requires_grad) {
    for (size_t i = 0; i < op_run_info->input_size; i++) {
      op_run_info->op_grad_info->input_value_grad_type[i] = PyNativeAlgo::Common::SetValueGradInfo(
        op_run_info->op_grad_info->input_value[i], nullptr, TensorGradType::kConstant);
      (void)op_run_info->base_op_run_info.input_tensor.emplace_back(
        op_run_info->op_grad_info->input_value[i]->cast<tensor::TensorPtr>());
    }
  }
  if (IsVmOp(op_run_info->base_op_run_info.op_name)) {
    std::vector<ValuePtr> result(op_run_info->input_size);
    for (size_t i = 0; i < op_run_info->input_size; i++) {
      result[i] = CopyTensorValueWithNewId(op_run_info->op_grad_info->input_value[i]);
    }
    auto result_v = ConstructOutputInVM(op_run_info, result);
    if (op_run_info->requires_grad) {
      (void)PyNativeAlgo::Common::SetValueGradInfo(result_v, nullptr, TensorGradType::kOpOutput);
    }
    MS_LOG(DEBUG) << "RunOpInVM end";
    return result_v;
  }

  MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info->op_prim);
  py::list vm_op_inputs = py::list(op_run_info->input_size);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    vm_op_inputs[i] = PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->op_grad_info->input_value[i]);
  }
  if (!utils::isa<PrimitivePy>(op_run_info->op_grad_info->op_prim)) {
    MS_LOG(EXCEPTION) << "Not a PrimitivePy, " << op_run_info->op_grad_info->op_prim->ToString();
  }
  auto result = utils::cast<PrimitivePyPtr>(op_run_info->op_grad_info->op_prim)->RunPyComputeFunction(vm_op_inputs);
  if (py::isinstance<py::none>(result)) {
    MS_LOG(EXCEPTION) << "VM op " << op_run_info->base_op_run_info.op_name << " run failed!";
  }
  ValuePtr result_v = PyNativeAlgo::DataConvert::PyObjToValue(result);
  if (!result_v->isa<ValueSequence>() && (op_run_info->base_op_run_info.abstract == nullptr ||
                                          op_run_info->base_op_run_info.abstract->isa<abstract::AbstractSequence>())) {
    result_v = std::make_shared<ValueTuple>(std::vector{result_v});
  }
  if (op_run_info->requires_grad) {
    (void)PyNativeAlgo::Common::SetValueGradInfo(result_v, nullptr, TensorGradType::kOpOutput);
  }
  MS_LOG(DEBUG) << "RunOpInVM end";
  return result_v;
}

void ForwardExecutor::CheckIfNeedSyncForHeterogeneous(const std::string &cur_target) {
  if (last_target_ != "Unknown" && last_target_ != cur_target) {
    Sync();
  }
  last_target_ = cur_target;
}

bool ForwardExecutor::CellNotSetMixedPrecision(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &cur_cell = forward_cell_stack_.top();
  MS_EXCEPTION_IF_NULL(cur_cell);
  MixedPrecisionType mix_type = cur_cell->GetMixedPrecisionType();
  if (mix_type == kNotSet) {
    return true;
  }
  op_run_info->mix_type = mix_type;
  return false;
}

void ForwardExecutor::ExecuteLazyTask() const {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kWaitPipeline);
  GilReleaseWithCheck gil_release;
  runtime::OpExecutor::GetInstance().WaitAll();
}

void ForwardExecutor::PrintPyObjInfo(const py::object &obj, const std::string &str, bool is_cell) const {
  if (is_cell) {
    MS_LOG(DEBUG) << str << " run " << obj.cast<CellPtr>()->ToString();
    return;
  }
  MS_LOG(DEBUG) << str << " run python function " << py::getattr(obj, "__name__").cast<std::string>();
}

void ForwardExecutor::ProcessBeforeNewGraph(const py::object &obj) {
  if (IsFirstCell()) {
    ReInit();
  }
  bool is_cell = py::isinstance<Cell>(obj);
  if (is_cell) {
    PushForwardCell(obj);
  }
  PrintPyObjInfo(obj, kBegin, is_cell);
  infer_operation()->set_only_single_op_run(false);
  if (!grad()->RequiresGrad()) {
    const auto &obj_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
    if (grad()->is_cell_has_dynamic_inputs(obj_id)) {
      MS_LOG(DEBUG) << "obj id:" << obj_id << " set forward use dynamic shape process true";
      grad()->set_forward_use_dynamic_shape_process(true);
#ifndef ENABLE_SECURITY
      ProfilerManager::GetInstance()->SetNetDynamicShapeStatus();
#endif
    }
  }
}

void ForwardExecutor::ProcessAfterNewGraph(const py::object &obj) const { grad()->SetTopCellDynamicAttr(obj); }

void ForwardExecutor::ProcessBeforeEndGraph(const py::object &obj, bool is_cell) {
  if (is_cell) {
    PopForwardCell();
  }
  if (!grad()->RequiresGrad()) {
    PrintPyObjInfo(obj, kEnd, is_cell);
  }

  // Do some finishing work before end graph
  if (IsFirstCell()) {
    if (frontend_queue_ != nullptr) {
      runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kWaitPipeline);
      GilReleaseWithCheck gil_release;
      frontend_queue_->Wait();
      backend_queue_->Wait();
    }
    // Finish lazy task
    ExecuteLazyTask();
    if (!grad()->RequiresGrad()) {
      ClearNodeAbsMap();
    }
    if (grad()->forward_use_dynamic_shape_process()) {
      MS_LOG(DEBUG) << "first cell run end, set forward use dynamic shape process false";
      grad()->set_forward_use_dynamic_shape_process(false);
    }
  }
}

void ForwardExecutor::ProcessAfterEndGraph(const py::object &obj, bool is_cell) const {
  if (IsFirstCell()) {
    auto forward_task =
      std::make_shared<FrontendTask>([this](const FrontendOpRunInfoPtr &op_run_info) { ClearNodeAbsMap(); }, nullptr);
    frontend_queue_->Push(forward_task);
  }
  PrintPyObjInfo(obj, kEnd, is_cell);
}

std::string ForwardExecutor::GetCurrentDeviceTarget(const PrimitivePtr &op_prim) const {
  MS_EXCEPTION_IF_NULL(op_prim);
  PrimitiveReadLock read_lock(op_prim->shared_mutex());
  const auto &attr_map = op_prim->attrs();
  auto iter = attr_map.find("primitive_target");
  if (iter != attr_map.end()) {
    return GetValue<std::string>(iter->second);
  }
  return device_target_;
}

void ForwardExecutor::Sync() {
  ExecuteLazyTask();

  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kSyncStream);
  for (auto &item : mindrt_backends_) {
    MS_EXCEPTION_IF_NULL(item.second);
    item.second->SyncStream();
  }
}

ValuePtr ForwardExecutor::RunOpInMs(const FrontendOpRunInfoPtr &op_run_info,
                                    const BackendOpRunInfoPtr &backend_op_run_info) {
  if (!ScopedFallbackRunning::on()) {
    GilReleaseWithCheck gil_relase;
    return RunOpInMsInner(op_run_info, backend_op_run_info);
  }
  // Print the op running in JIT Fallback.
  static const auto dump_fallback = (common::GetEnv("MS_DEV_FALLBACK_DUMP_NODE") == "1");
  if (dump_fallback) {
    MS_LOG(ERROR) << "NOTICE: The op is running in JIT Fallback:\n"
                  << "primitive: " << op_run_info->op_grad_info->op_prim->ToString();
  } else {
    MS_LOG(INFO) << "NOTICE: The op is running in JIT Fallback:\n"
                 << "primitive: " << op_run_info->op_grad_info->op_prim->ToString();
  }
  return RunOpInMsInner(op_run_info, backend_op_run_info);
}

void ForwardExecutor::PrepareOpInputs(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  CheckIfNeedSyncForHeterogeneous(op_run_info->base_op_run_info.device_target);
  PyNativeAlgo::DataConvert::GetInputTensor(op_run_info, op_run_info->base_op_run_info.device_target,
                                            op_run_info->requires_grad ? grad()->top_cell() : nullptr);
}

void ForwardExecutor::PrepareOpOutputs(const FrontendOpRunInfoPtr &op_run_info) {
  CreateOutputTensor(op_run_info->base_op_run_info.abstract, &op_run_info->output_tensors,
                     &op_run_info->device_sync_promises);
  TransformOutputValues(op_run_info);
  UpdateOutputStubNodeValue(op_run_info, op_run_info->real_out);
  // Not use GetNext abs
  if (op_run_info->base_op_run_info.op_name != kGetNextOpName) {
    op_run_info->out_value_id = PyNativeAlgo::Common::GetIdByValue(op_run_info->real_out);
    // save abs for next infer
    SetNodeAbsMapByValue(op_run_info);
  }
}

ValuePtr ForwardExecutor::RunOpInMsInner(const FrontendOpRunInfoPtr &op_run_info,
                                         const BackendOpRunInfoPtr &backend_op_run_info) {
  const auto &outputs = RunOpBackendInner(op_run_info, backend_op_run_info);
  bool is_out_sequence = (op_run_info->base_op_run_info.abstract == nullptr ||
                          op_run_info->base_op_run_info.abstract->isa<abstract::AbstractSequence>());
  const auto &result_v =
    PyNativeAlgo::DataConvert::VectorRefToValue(outputs, op_run_info->requires_grad, is_out_sequence);
  MS_LOG(DEBUG) << "RunOpInMs end";
  return result_v;
}

void ForwardExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear forward res";
  {
    GilReleaseWithCheck gil_release;
    frontend_queue_->Clear();
    backend_queue_->Clear();
  }
  for (const auto &item : mindrt_backends_) {
    MS_EXCEPTION_IF_NULL(item.second);
    item.second->ClearOpExecutorResource();
  }
  init_ = false;
  lazy_build_ = false;
  is_ms_function_compiling_ = false;
  cast_operation()->ClearRes();
  ClearNodeAbsMap();
  infer_operation()->ClearPrimAbsList();
  infer_operation()->ClearConstFlagPrimCache();
  std::stack<CellPtr>().swap(forward_cell_stack_);
  mindrt_backends_.clear();
  ForkUtils::GetInstance().DeregCallbacks(this);
}

void ForwardExecutor::ReinitAfterFork() {
  MS_LOG(INFO) << "fork event detected in child process, ForwardExecutor resources will be reinitialized.";
  // reset ms context after fork
  MsContext::GetInstance()->ResetContext();
  // clear op cache after fork
  OpCompiler::GetInstance().ClearAllCache();
  // clear backend caches
  for (const auto &item : mindrt_backends_) {
    MS_EXCEPTION_IF_NULL(item.second);
    item.second->ClearOpExecutorResource();
  }
  mindrt_backends_.clear();
}
}  // namespace pynative
}  // namespace mindspore
