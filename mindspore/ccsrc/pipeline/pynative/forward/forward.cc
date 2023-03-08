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
#include <vector>
#include "pipeline/pynative/pynative_utils.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/python_fallback_running.h"
#include "backend/graph_compiler/transform.h"
#include "utils/ms_context.h"
#include "pipeline/pynative/forward/forward_task.h"
#include "include/common/utils/stub_tensor.h"
#include "runtime/pynative/op_executor.h"

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

void GetSingleOpGraphInfo(const FrontendOpRunInfoPtr &op_run_info, const std::string &cur_target) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const std::vector<tensor::TensorPtr> &input_tensors = op_run_info->base_op_run_info.input_tensor;
  const std::vector<int64_t> &tensors_mask = op_run_info->base_op_run_info.input_mask;
  if (input_tensors.size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  std::ostringstream buf;
  buf << cur_target << "_dynamic" << op_run_info->base_op_run_info.use_dynamic_shape_process << "_";
  buf << op_run_info->base_op_run_info.op_name << "_";
  const auto &op_prim = op_run_info->op_prim;
  MS_EXCEPTION_IF_NULL(op_prim);
  bool has_hidden_side_effect = op_prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_HIDDEN);
  for (size_t index = 0; index < input_tensors.size(); ++index) {
    const auto &input_tensor = input_tensors[index];
    MS_EXCEPTION_IF_NULL(input_tensor);
    bool use_dynamic_shape_process = op_run_info->base_op_run_info.use_dynamic_shape_process;
    if (use_dynamic_shape_process) {
      buf << input_tensor->shape().size() << "_";
    } else {
      if (input_tensor->base_shape_ptr() != nullptr) {
        buf << input_tensor->base_shape_ptr()->ToString();
      } else {
        buf << input_tensor->shape();
      }
    }
    buf << input_tensor->data_type();
    buf << input_tensor->padding_type();
    // In the case of the same shape, but dtype and format are inconsistent
    auto tensor_addr = input_tensor->device_address();
    if (tensor_addr != nullptr && !has_hidden_side_effect) {
      auto p_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor_addr);
      MS_EXCEPTION_IF_NULL(p_address);
      buf << p_address->type_id();
      buf << p_address->format();
    }
    // For constant input
    if (tensors_mask[index] == kValueNodeTensorMask) {
      buf << common::AnfAlgo::GetTensorValueString(input_tensor);
    }
    buf << "_";
  }
  // The value of the attribute affects the operator selection
  const auto &attr_map = op_prim->attrs();
  (void)std::for_each(attr_map.begin(), attr_map.end(),
                      [&buf](const auto &element) { buf << element.second->ToString(); });

  // Operator with hidden side effect.
  if (has_hidden_side_effect) {
    buf << "_" << std::to_string(op_prim->id());
  }
  op_run_info->base_op_run_info.graph_info = buf.str();
}
}  // namespace

std::string ForwardExecutor::device_target() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  return ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
}

void ForwardExecutor::WaitForwardTask() {
  GilReleaseWithCheck gil_release;
  if (forward_queue_ != nullptr) {
    forward_queue_->Wait();
  }
}

bool ForwardExecutor::IsVmOp(const std::string &op_name) const {
  return kVmOperators.find(op_name) != kVmOperators.end();
}

GradExecutorPtr ForwardExecutor::grad() const {
  auto grad_executor = grad_executor_.lock();
  MS_EXCEPTION_IF_NULL(grad_executor);
  return grad_executor;
}

void ForwardExecutor::Init() {
  if (init_) {
    return;
  }
  MS_LOG(DEBUG) << "Init ForwardExecutor";
  compile::SetMindRTEnable();
  python_adapter::set_python_env_flag(true);
  init_ = true;
  runtime::OpExecutor::GetInstance().RegisterForwardCallback([this]() { forward_queue_->Wait(); });
}

void ForwardExecutor::RunOpForwardAsyncImpl(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "RunOp name: " << op_run_info->base_op_run_info.op_name;
  PyNativeAlgo::Common::StubNodeToValue(op_run_info);
  // 1.Set cast for inputs
  SetCastForInputs(op_run_info);
  // 2.Infer output abstract
  InferOutputAbstract(op_run_info);
  // Update stub node abstract
  const auto &abs = op_run_info->base_op_run_info.abstract;
  auto success = op_run_info->stub_output->SetAbstract(abs);
  if (!success) {
    MS_EXCEPTION(TypeError) << "The predict type and infer type is not match, infer type is " << abs->BuildType()
                            << ", the name of operator is [" << op_run_info->base_op_run_info.op_name
                            << "]. Please modify or add predict type of operator in predict_out_type_map.h.";
  }

  // 3.Run op with selected backend
  if (!op_run_info->output_get_by_infer_value) {
    GetOutput(op_run_info);
  }

  op_run_info->stub_output->SetValue(op_run_info->out_value);

  if (!op_run_info->grad_flag) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }

  // Const value no need do op grad
  if (op_run_info->output_get_by_infer_value) {
    return;
  }
  // 4. Do op grad and record op info
  // If ms function is compile, op info will not be find in second training step
  if (!op_run_info->async_status.is_ms_function_compiling && grad()->custom_bprop_cell_count() <= 0) {
    grad()->ProcessOpGradInfo(op_run_info);
  }
}

void ForwardExecutor::RunOpForwardAsync(const FrontendOpRunInfoPtr &op_run_info) {
  Init();
  auto forward_task = std::make_shared<ForwardTask>(
    [this](const FrontendOpRunInfoPtr &op_run_info) { RunOpForwardAsyncImpl(op_run_info); }, op_run_info);
  forward_queue_->Push(forward_task);
}

void ForwardExecutor::RunOpForward(const FrontendOpRunInfoPtr &op_run_info) {
  Init();
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "RunOp name: " << op_run_info->base_op_run_info.op_name;
  // 1.Set cast for inputs
  SetCastForInputs(op_run_info);
  // 2.Infer output abstract
  InferOutputAbstract(op_run_info);
  // 3.Run op with selected backend
  if (!op_run_info->output_get_by_infer_value) {
    GetOutput(op_run_info);
  }
  if (!op_run_info->grad_flag) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  // Const value no need do op grad
  if (op_run_info->output_get_by_infer_value) {
    return;
  }

  // 4. Do op grad and record op info
  // If ms function is compile, op info will not be find in second training step
  if (!op_run_info->async_status.is_ms_function_compiling && grad()->custom_bprop_cell_count() <= 0) {
    grad()->ProcessOpGradInfo(op_run_info);
  }
}

FrontendOpRunInfoPtr ForwardExecutor::GenerateOpRunInfo(const py::args &args, bool stub) const {
  if (args.size() != static_cast<size_t>(RunOpArgsEnum::PY_ARGS_NUM)) {
    MS_LOG(EXCEPTION) << "Three args are needed by RunOp";
  }
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  // Used for async run
  op_run_info->grad_flag = grad()->grad_flag();
  if (op_run_info->grad_flag) {
    op_run_info->base_op_run_info.use_dynamic_shape_process = grad()->use_dynamic_shape_process();
  } else {
    op_run_info->base_op_run_info.use_dynamic_shape_process = grad()->forward_use_dynamic_shape_process();
  }
  op_run_info->base_op_run_info.op_name = args[static_cast<size_t>(RunOpArgsEnum::PY_NAME)].cast<std::string>();
  op_run_info->base_op_run_info.lazy_build = lazy_build_;
  PyNativeAlgo::PyParser::SetPrim(op_run_info, args[static_cast<size_t>(RunOpArgsEnum::PY_PRIM)]);
  PyNativeAlgo::PyParser::ParseOpInputByPythonObj(op_run_info, args[static_cast<size_t>(RunOpArgsEnum::PY_INPUTS)],
                                                  stub);
  (void)op_run_prim_py_list_.emplace_back(op_run_info->op_prim);
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

const NodeAbsCache &ForwardExecutor::NodeAbsMap() const { return infer_operation()->node_abs_cache(); }

void ForwardExecutor::InferOutputAbstract(const FrontendOpRunInfoPtr &op_run_info) const {
  infer_operation()->DoInfer(op_run_info);
}

void ForwardExecutor::GetOutput(const FrontendOpRunInfoPtr &op_run_info) {
  // Run op with selected backend, nop is no need run backend
  op_run_info->out_value = RunOpWithBackendPolicy(op_run_info);
  if (op_run_info->out_value->isa<ValueSequence>()) {
    const auto &result_v_list = op_run_info->out_value->cast<ValueSequencePtr>();
    if (result_v_list->size() == 1 && op_run_info->base_op_run_info.abstract != nullptr &&
        !op_run_info->base_op_run_info.abstract->isa<abstract::AbstractSequence>()) {
      op_run_info->out_value = result_v_list->value().front();
    }
  }

  // Not use GetNext abs
  if (op_run_info->base_op_run_info.op_name != kGetNextOpName) {
    op_run_info->out_value_id = PyNativeAlgo::Common::GetIdByValue(op_run_info->out_value);
    SetNodeAbsMapByValue(op_run_info);
  }
}

compile::MindRTBackendPtr ForwardExecutor::GetMindRtBackend(const std::string &device_target) {
  auto iter = mindrt_backends_.find(device_target);
  if (iter == mindrt_backends_.end()) {
    std::lock_guard<std::mutex> guard(pipeline::Resource::GetBackendInitMutex());
    auto backend = std::make_shared<compile::MindRTBackend>("ms", device_target, device_id_);
    MS_EXCEPTION_IF_NULL(backend);
    mindrt_backends_[device_target] = backend;
    return backend;
  } else {
    return iter->second;
  }
}

ValuePtr ForwardExecutor::RunOpWithBackendPolicy(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  ValuePtr result;
  auto backend_policy = GetBackendPolicy(device_target());
  if (backend_policy == kMsBackendVmOnly) {
#ifndef ENABLE_TEST
    if (IsVmOp(op_run_info->base_op_run_info.op_name)) {
      result = RunOpInVM(op_run_info);
    } else {
      result = RunOpInMs(op_run_info);
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
  MS_EXCEPTION_IF_NULL(op_run_info->op_prim);
  op_run_info->run_in_vm = true;
  if (IsVmOp(op_run_info->base_op_run_info.op_name)) {
    std::vector<ValuePtr> result(op_run_info->input_size);
    for (size_t i = 0; i < op_run_info->input_size; i++) {
      bool input_is_tensor = op_run_info->input_value[i]->isa<tensor::Tensor>();
      if (input_is_tensor && op_run_info->base_op_run_info.op_name != prim::kPrimHookBackward->name() &&
          op_run_info->base_op_run_info.op_name != prim::kPrimCellBackwardHook->name()) {
        auto tensor = op_run_info->input_value[i]->cast<tensor::TensorPtr>();
        // Just get new tensor id
        auto new_tensor = std::make_shared<tensor::Tensor>(tensor->data_type(), tensor->shape(), tensor->data_ptr());
        new_tensor->set_device_address(tensor->device_address());
        new_tensor->set_sync_status(tensor->sync_status());
        new_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
        result[i] = new_tensor;
      } else {
        result[i] = op_run_info->input_value[i];
      }
    }
    auto result_v = std::make_shared<ValueTuple>(result);
    MS_LOG(DEBUG) << "RunOpInVM end";
    return result_v;
  }

  MS_EXCEPTION_IF_NULL(op_run_info->op_prim);
  py::list vm_op_inputs = py::list(op_run_info->input_size);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    vm_op_inputs[i] = PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->input_value[i]);
  }
  if (!utils::isa<PrimitivePy>(op_run_info->op_prim)) {
    MS_LOG(EXCEPTION) << "Not a PrimitivePy, " << op_run_info->op_prim->ToString();
  }
  auto result = utils::cast<PrimitivePyPtr>(op_run_info->op_prim)->RunPyComputeFunction(vm_op_inputs);
  if (py::isinstance<py::none>(result)) {
    MS_LOG(EXCEPTION) << "VM op " << op_run_info->base_op_run_info.op_name << " run failed!";
  }
  ValuePtr result_v = PyNativeAlgo::DataConvert::PyObjToValue(result);
  MS_LOG(DEBUG) << "RunOpInVM end";
  if (result_v->isa<ValueSequence>()) {
    return result_v;
  }
  return std::make_shared<ValueTuple>(std::vector{result_v});
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

void ForwardExecutor::ExecuteLazyTask() {
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

void ForwardExecutor::ProcessBeforeNewGraph(const py::object &obj, const py::args &args) {
  bool is_cell = py::isinstance<Cell>(obj);
  if (is_cell) {
    PushForwardCell(obj);
  }
  PrintPyObjInfo(obj, kBegin, is_cell);
  infer_operation()->set_only_single_op_run(false);
  if (!grad()->grad_flag()) {
    const auto &obj_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
    if (grad()->is_cell_has_dynamic_inputs(obj_id)) {
      MS_LOG(DEBUG) << "obj id:" << obj_id << " set forward use dynamic shape process true";
      grad()->set_forward_use_dynamic_shape_process(true);
    }
  }
}

void ForwardExecutor::ProcessAfterNewGraph(const py::object &obj) { grad()->SetTopCellDynamicAttr(obj); }

void ForwardExecutor::ProcessBeforeEndGraph(const py::object &obj, bool is_cell) {
  if (is_cell) {
    PopForwardCell();
  }
  if (!grad()->grad_flag()) {
    PrintPyObjInfo(obj, kEnd, is_cell);
  }

  // Do some finishing work before end graph
  if (IsFirstCell()) {
    if (forward_queue_ != nullptr) {
      GilReleaseWithCheck gil_release;
      forward_queue_->Wait();
    }
    // Finish lazy task
    ExecuteLazyTask();
    if (!grad()->grad_flag()) {
      ClearNodeAbsMap();
    }
  }
}

void ForwardExecutor::ProcessAfterEndGraph(const py::object &obj, bool is_cell) const {
  if (IsFirstCell()) {
    ClearNodeAbsMap();
    if (!grad()->grad_flag()) {
      MS_LOG(DEBUG) << "first cell run end, set forward use dynamic shape process false";
      grad()->set_forward_use_dynamic_shape_process(false);
    }
  }
  PrintPyObjInfo(obj, kEnd, is_cell);
}

std::string ForwardExecutor::GetCurrentDeviceTarget(const PrimitivePtr &op_prim) {
  MS_EXCEPTION_IF_NULL(op_prim);
  const auto &attr_map = op_prim->attrs();
  auto iter = attr_map.find("primitive_target");
  if (iter != attr_map.end()) {
    return GetValue<std::string>(iter->second);
  }
  return device_target();
}

void ForwardExecutor::Sync() {
  ExecuteLazyTask();

  for (auto &item : mindrt_backends_) {
    MS_EXCEPTION_IF_NULL(item.second);
    item.second->SyncStream();
  }
  {
    py::gil_scoped_acquire acquire;
    op_run_prim_py_list_.clear();
  }
}

ValuePtr ForwardExecutor::RunOpInMs(const FrontendOpRunInfoPtr &op_run_info) {
  if (!ScopedFallbackRunning::on()) {
    GilReleaseWithCheck gil_relase;
    return RunOpInMsInner(op_run_info);
  }
  return RunOpInMsInner(op_run_info);
}

ValuePtr ForwardExecutor::RunOpInMsInner(const FrontendOpRunInfoPtr &op_run_info) {
  MS_LOG(DEBUG) << "RunOpInMs start";
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_id_ = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, true);
  const auto &cur_target = GetCurrentDeviceTarget(op_run_info->op_prim);
  op_run_info->base_op_run_info.device_target = cur_target;
  CheckIfNeedSyncForHeterogeneous(cur_target);
  PyNativeAlgo::DataConvert::GetInputTensor(op_run_info, cur_target);
  // get graph info for checking it whether existing in the cache
  GetSingleOpGraphInfo(op_run_info, cur_target);
  auto backend_op_run_info = std::make_shared<BackendOpRunInfo>(
    op_run_info->base_op_run_info, std::make_shared<Primitive>(*op_run_info->op_prim), true, false);

#if defined(__APPLE__)
  backend_op_run_info->base_op_run_info.lazy_build = false;
#endif

  VectorRef outputs;
  const auto &cur_mind_rt_backend = GetMindRtBackend(cur_target);
  MS_EXCEPTION_IF_NULL(cur_mind_rt_backend);
  bool use_dynamic_shape_process = op_run_info->base_op_run_info.use_dynamic_shape_process;
  if (use_dynamic_shape_process) {
    AnfAlgo::SetDynamicAttrToPrim(backend_op_run_info->op_prim);
    cur_mind_rt_backend->RunOpDynamic(backend_op_run_info, &outputs);
  } else {
    cur_mind_rt_backend->RunOp(backend_op_run_info, &outputs);
  }

  if (op_run_info->base_op_run_info.has_dynamic_output) {
    op_run_info->base_op_run_info.abstract = backend_op_run_info->base_op_run_info.abstract;
  }
  const auto &result_v = PyNativeAlgo::DataConvert::VectorRefToValue(outputs);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  MS_LOG(DEBUG) << "RunOpInMs end";
  return result_v;
}

void ForwardExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear forward res";
  {
    GilReleaseWithCheck gil_release;
    forward_queue_->Clear();
  }
  for (const auto &item : mindrt_backends_) {
    MS_EXCEPTION_IF_NULL(item.second);
    item.second->ClearOpExecutorResource();
  }
  init_ = false;
  lazy_build_ = false;
  is_ms_function_compiling_ = false;
  ClearNodeAbsMap();
  infer_operation()->ClearPrimAbsList();
  infer_operation()->ClearConstFlagPrimCache();
  std::stack<CellPtr>().swap(forward_cell_stack_);
  mindrt_backends_.clear();
  op_run_prim_py_list_.clear();
}
}  // namespace pynative
}  // namespace mindspore
