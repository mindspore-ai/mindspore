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
#include "ops/structure_op_name.h"
#include "ops/conv_pool_op_name.h"
#include "ops/nn_op_name.h"
#include "ops/array_ops.h"
#include "ops/framework_ops.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/python_fallback_running.h"
#include "backend/graph_compiler/transform.h"
#include "utils/ms_context.h"
#include "pipeline/pynative/forward/forward_task.h"
#include "pipeline/pynative/predict_out_type_map.h"
#include "include/common/utils/stub_tensor.h"
#include "runtime/pynative/op_executor.h"
#ifndef ENABLE_SECURITY
#include "include/backend/debug/profiler/profiling.h"
using mindspore::profiler::ProfilerManager;
#endif
#include "include/common/utils/tensor_future.h"
#include "frontend/operator/ops_front_infer_function.h"

namespace mindspore {
namespace pynative {
namespace {
const mindspore::HashMap<std::string, mindspore::HashMap<size_t, std::string>> kSliceOpInputToAttr = {
  {kBroadcastToOpName, {{0, ops::kShape}}}};
const std::set<std::string> kVmOperators = {"InsertGradientOf", "StopGradient", "HookBackward", "CellBackwardHook"};
const std::set<std::string> kViewOpForComplexToOtherType = {"Real", "Imag"};
constexpr char kBegin[] = "Begin";
constexpr char kEnd[] = "End";
constexpr auto kOpNameCustom = "Custom";
enum class RunOpArgsEnum : size_t { PY_PRIM = 0, PY_NAME, PY_INPUTS, PY_ARGS_NUM };

// Shallow Copy Value and change shape
ValuePtr ShallowCopyValue(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value);
  auto tensor_abs = op_run_info->base_op_run_info.abstract;
  MS_EXCEPTION_IF_NULL(tensor_abs);
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
  } else if (v->isa<ValueTuple>()) {
    const auto &v_tup = v->cast<ValueTuplePtr>();
    ValuePtrList list;
    for (const auto &ele : v_tup->value()) {
      (void)list.emplace_back(CopyTensorValueWithNewId(ele));
    }
    return std::make_shared<ValueTuple>(list);
  } else if (v->isa<ValueList>()) {
    const auto &v_list = v->cast<ValueListPtr>();
    ValuePtrList list;
    for (const auto &ele : v_list->value()) {
      (void)list.emplace_back(CopyTensorValueWithNewId(ele));
    }
    return std::make_shared<ValueList>(list);
  } else {
    return v;
  }
}

void UpdateOutputStubNodeAbs(const FrontendOpRunInfoPtr &op_run_info) {
  if (op_run_info->stub_output == nullptr) {
    return;
  }
  const auto &abs = op_run_info->base_op_run_info.abstract;
  MS_EXCEPTION_IF_NULL(abs);
  auto success = op_run_info->stub_output->SetAbstract(abs);
  if (!success) {
    const auto &op_name = op_run_info->base_op_run_info.op_name;
    MS_EXCEPTION(TypeError) << "The predict type and infer type is not match, predict type is "
                            << PredictOutType(op_run_info) << ", infer type is " << abs->BuildType()
                            << ", the name of operator is [" << op_name
                            << "]. Please modify or add predict type of operator in predict_out_type_map.h.";
  }
  MS_LOG(DEBUG) << "Update StubNode abstract " << abs->ToString();
}

void ClonePrim(const FrontendOpRunInfoPtr &op_run_info) {
  // Clone a new prim
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto prim = op_run_info->op_grad_info->op_prim;
  auto prim_py = prim->cast<PrimitivePyPtr>();
  if (prim_py == nullptr) {
    return;
  }
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
  if (result.size() == 1) {
    return result[kIndex0];
  }
  return std::make_shared<ValueTuple>(result);
}

void UpdateOutputStubNodeValue(const FrontendOpRunInfoPtr &op_run_info) {
  if (op_run_info->stub_output != nullptr) {
    op_run_info->stub_output->SetValue(op_run_info->real_out);
  }
}

BackendOpRunInfoPtr CreateBackendOpRunInfo(const FrontendOpRunInfoPtr &op_run_info) {
  auto backend_op_run_info = std::make_shared<BackendOpRunInfo>(
    op_run_info->base_op_run_info, std::make_shared<Primitive>(*op_run_info->op_grad_info->op_prim), true, false);
  // Need to update promise in backend task.
  backend_op_run_info->device_sync_promises = std::move(op_run_info->device_sync_promises);
  // Erase RandomOp cache avoid memory leak.
  if (AnfAlgo::NeedEraseCache(backend_op_run_info->op_prim)) {
    op_run_info->base_op_run_info.need_earse_cache = true;
  }
  if (op_run_info->base_op_run_info.has_dynamic_output && op_run_info->base_op_run_info.op_name != kGetNextOpName) {
    backend_op_run_info->base_op_run_info.use_dynamic_shape_process = true;
  }
  return backend_op_run_info;
}

void TransformOutputValues(const FrontendOpRunInfoPtr &op_run_info) {
  std::vector<ValuePtr> output_values;
  for (auto &output_tensor : op_run_info->base_op_run_info.output_tensors) {
    MS_EXCEPTION_IF_NULL(output_tensor);

    if (op_run_info->requires_grad) {
      output_tensor->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>());
      output_tensor->auto_grad_meta_data()->set_grad_type(TensorGradType::kOpOutput);
    }
    (void)output_values.emplace_back(output_tensor);
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
    (void)outputs->emplace_back(output_tensor);
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
  return !OpCompiler::GetInstance().IsInvalidInferResultOp(op_run_info->base_op_run_info.op_name) &&
         !op_run_info->base_op_run_info.has_dynamic_output;
}

KernelTaskType GetViewOpTaskType(const std::string &op_name) {
  if (op_name == kCopyWithScileOpName) {
    return KernelTaskType::kCOPY_TASK;
  }
  return KernelTaskType::kNORMAL_VIEW_TASK;
}

void EmplaceSliceInputs(const FrontendOpRunInfoPtr &op_run_info, const std::vector<ValuePtr> &input_values,
                        const SliceOpInfoPtr &slice_op_info) {
  for (auto idx : slice_op_info->data_indexs) {
    if (idx >= input_values.size()) {
      MS_LOG(EXCEPTION) << "data_idx is out of bounds, data_idx:" << idx
                        << " input_values.size():" << input_values.size();
    }
    (void)op_run_info->op_grad_info->input_value.emplace_back(input_values[idx]);
  }

  mindspore::HashMap<size_t, std::string> input_to_attr;
  auto iter = kSliceOpInputToAttr.find(op_run_info->base_op_run_info.op_name);
  if (iter != kSliceOpInputToAttr.end()) {
    input_to_attr = iter->second;
  }

  for (size_t i = 0; i < slice_op_info->slice_index_inputs.size(); i++) {
    auto slice_index = slice_op_info->slice_index_inputs[i];
    ValuePtr v = nullptr;
    if (slice_index->is_int()) {
      v = MakeValue(slice_index->int_value());
    } else {
      v = MakeValue(slice_index->vec_value());
    }

    auto idx_iter = input_to_attr.find(i);
    if (idx_iter == input_to_attr.end()) {
      (void)op_run_info->op_grad_info->input_value.emplace_back(v);
      continue;
    }

    MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info->op_prim);
    const auto &attr_name = idx_iter->second;
    op_run_info->op_grad_info->op_prim->set_attr(attr_name, v);
  }

  op_run_info->input_size = op_run_info->op_grad_info->input_value.size();
  PyNativeAlgo::PyParser::PrepareOpGradInfo(op_run_info);
}

bool EnableView(const FrontendOpRunInfoPtr &op_run_info) {
  if (op_run_info->base_op_run_info.device_target != kAscendDevice) {
    return true;
  }

  if (op_run_info->op_grad_info->input_value.empty()) {
    MS_LOG(EXCEPTION) << "View, op:" << op_run_info->base_op_run_info.op_name << " input_value is empty.";
  }

  auto view_value = op_run_info->op_grad_info->input_value[0];
  MS_EXCEPTION_IF_NULL(view_value);
  if (!view_value->isa<tensor::Tensor>()) {
    MS_EXCEPTION(TypeError) << "input value is not Tensor";
  }
  auto tensor = view_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &type_id = tensor->data_type();
  if (type_id == kNumberTypeComplex128 || type_id == kNumberTypeFloat64) {
    // AsStrided and ViewCopy is not support Complex128 and Float64, disable view
    MS_LOG(DEBUG) << "Disable view, op:" << op_run_info->base_op_run_info.op_name
                  << " device_target:" << op_run_info->base_op_run_info.device_target
                  << " type:" << TypeIdToString(type_id);
    return false;
  }

  return true;
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
  // cppcheck-suppress unreadVariable
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
}

void ForwardExecutor::RefreshForwardCallback() {
#if defined(_WIN32) || defined(_WIN64)
  runtime::OpExecutor::GetInstance().RegisterForwardCallback([this]() {
    frontend_queue_->Wait();
    backend_queue_->Wait();
    grad()->WaitBpropTask();
  });
#endif
  // ForwardCallback has been set in ForwardExecutor::Init, no need to refresh anymore.
  return;
}

bool ForwardExecutor::enable_async() const {
#if defined(ENABLE_TEST) || defined(__APPLE__)
  return false;
#else
  return enable_async_;
#endif
}

bool ForwardExecutor::EnablePipeline(const std::string &op_name) const {
  return enable_async() && !IsVmOp(op_name) && op_name != kOpNameCustom && !ScopedFallbackRunning::on() &&
         MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode;
}

void ForwardExecutor::DispatchFrontendTask(const FrontendOpRunInfoPtr &op_run_info) {
  auto forward_task = std::make_shared<FrontendTask>(
    [this](const FrontendOpRunInfoPtr &op_run_info) { RunOpFrontend(op_run_info); }, op_run_info);
  frontend_queue_->Push(forward_task);
}

void ForwardExecutor::ForwardOpGradImpl(const FrontendOpRunInfoPtr &op_run_info) {
  if (!op_run_info->requires_grad) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  // 4. Do op grad and record op info
  // If ms function is compile, op info will not be find in second training step
  if (!op_run_info->async_status.is_jit_compiling && op_run_info->async_status.custom_bprop_cell_count <= 0) {
    grad()->ProcessOpGradInfo(op_run_info);
  }
}

void ForwardExecutor::ForwardRunViewKernelTask(const FrontendOpRunInfoPtr &op_run_info, const KernelTaskType &task_type,
                                               bool enable_async) {
  if (task_type == KernelTaskType::kNORMAL_VIEW_TASK) {
    return;
  }
  MS_LOG(DEBUG) << "Start, task_type:" << task_type;

  const auto &cur_mind_rt_backend = GetMindRtBackend(op_run_info->base_op_run_info.device_target);
  MS_EXCEPTION_IF_NULL(cur_mind_rt_backend);
  cur_mind_rt_backend->RunViewKernelTask(op_run_info->base_op_run_info, task_type, enable_async);

  MS_LOG(DEBUG) << "End";
}

void ForwardExecutor::DispatchViewKernelTask(const FrontendOpRunInfoPtr &op_run_info, const KernelTaskType &task_type) {
  static auto run_backend_with_grad = [this](const FrontendOpRunInfoPtr &op_run_info, const KernelTaskType &task_type) {
    ForwardRunViewKernelTask(op_run_info, task_type, true);
    ForwardOpGradImpl(op_run_info);
  };

  auto backend_task = std::make_shared<ViewKernelBackendTask>(run_backend_with_grad, op_run_info, task_type);
  backend_queue_->Push(backend_task);
}  // namespace pynative

void ForwardExecutor::DispatchBackendTask(const FrontendOpRunInfoPtr &op_run_info,
                                          const session::BackendOpRunInfoPtr &backend_op_run_info) {
  static auto run_backend_with_grad = [this](const FrontendOpRunInfoPtr &op_run_info,
                                             const session::BackendOpRunInfoPtr &backend_op_run_info) {
    // Update tensor device address in backend.
    RunOpBackendInner(op_run_info, backend_op_run_info);
    ForwardOpGradImpl(op_run_info);
  };

  auto backend_task = std::make_shared<BackendTask>(run_backend_with_grad, op_run_info, backend_op_run_info);
  backend_queue_->Push(backend_task);
}

void ForwardExecutor::CreateDeviceAddressForViewInput(const FrontendOpRunInfoPtr &op_run_info,
                                                      const tensor::TensorPtr &input_tensor, const size_t &input_idx,
                                                      bool enable_async, bool need_wait) {
  if (need_wait) {
    auto device_address = input_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->set_is_view(true);
    return;
  }
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {op_run_info->base_op_run_info.device_target, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  auto address_size = GetTypeByte(TypeIdToType(input_tensor->data_type())) * SizeOf(input_tensor->shape());
  auto device_address = device_context->device_res_manager_->CreateDeviceAddress(
    nullptr, address_size, kOpFormat_DEFAULT, input_tensor->data_type(), input_tensor->shape());
  device_address->set_is_view(true);
  if (!op_run_info->device_sync_promises.empty()) {
    MS_LOG(DEBUG) << "Has promise and update tensor address.";
    const auto &output_promise = op_run_info->device_sync_promises[input_idx];
    output_promise->SetValue(std::make_shared<pynative::DeviceAddressFutureData>(device_address, nullptr));
  } else {
    input_tensor->set_device_address(device_address);
  }

  const auto &cur_mind_rt_backend = GetMindRtBackend(op_run_info->base_op_run_info.device_target);
  MS_EXCEPTION_IF_NULL(cur_mind_rt_backend);
  cur_mind_rt_backend->RunAllocMemTask(device_context, input_tensor, enable_async);
}

void ForwardExecutor::RunContiguousTask(const tensor::TensorPtr &tensor, bool enable_async) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->storage_info() == nullptr) {
    return;
  }

  MS_LOG(DEBUG) << "Tensor storage_info is not nullptr, id:" << tensor->id();
  MS_EXCEPTION_IF_NULL(tensor->device_address());
  auto device_addr = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  const auto &cur_mind_rt_backend = GetMindRtBackend(device_addr->device_name());
  MS_EXCEPTION_IF_NULL(cur_mind_rt_backend);
  cur_mind_rt_backend->RunContiguousTask(tensor, enable_async);
}

TypePtr InferTypeForViewComplex(const tensor::TensorPtr &tensor) {
  if (tensor->data_type() == kNumberTypeComplex64) {
    return TypeIdToType(kNumberTypeFloat32);
  } else if (tensor->data_type() == kNumberTypeComplex128) {
    return TypeIdToType(kNumberTypeFloat64);
  } else {
    MS_LOG(EXCEPTION) << "tensor->data_type() is " << TypeIdToString(tensor->data_type()) << " unsupported";
  }
}

void ForwardExecutor::CreateViewOpOutputs(
  const FrontendOpRunInfoPtr &op_run_info, const tensor::TensorPtr &view_input_tensor,
  const TensorStorageInfoPtrList &storage_infos,
  const std::shared_ptr<tensor::FutureBase<DeviceSync>> &input_origin_address_future,
  const DeviceSyncPtr &input_origin_device_address, bool is_tuple_output) {
  TypePtr data_type = view_input_tensor->Dtype();
  if (kViewOpForComplexToOtherType.find(op_run_info->base_op_run_info.op_name) != kViewOpForComplexToOtherType.end()) {
    data_type = InferTypeForViewComplex(view_input_tensor);
  }
  // Generate output abs by storage_info.
  if (storage_infos.size() == 1 && !is_tuple_output) {
    op_run_info->base_op_run_info.abstract =
      abstract::MakeAbstractTensor(std::make_shared<abstract::Shape>(storage_infos[0]->shape), data_type);
    storage_infos[0]->data_type = data_type->type_id();
  } else {
    AbstractBasePtrList abs_list;
    for (const auto &storage_info : storage_infos) {
      auto abs = abstract::MakeAbstractTensor(std::make_shared<abstract::Shape>(storage_info->shape), data_type);
      storage_info->data_type = data_type->type_id();
      (void)abs_list.emplace_back(abs);
    }
    op_run_info->base_op_run_info.abstract = std::make_shared<abstract::AbstractTuple>(abs_list);
  }
  UpdateOutputStubNodeAbs(op_run_info);
  CreateInputAddressForViewOp(view_input_tensor, op_run_info, 0);

  for (size_t i = 0; i < storage_infos.size(); i++) {
    MS_LOG(INFO) << "View op " << op_run_info->base_op_run_info.op_name << ", i:" << i
                 << ", storage_info:" << storage_infos[i]->ToString();
    CreateViewOutputTensor(op_run_info, view_input_tensor, storage_infos[i], input_origin_address_future,
                           input_origin_device_address, data_type);
  }

  if (op_run_info->base_op_run_info.output_tensors.size() == 1 && !is_tuple_output) {
    op_run_info->real_out = op_run_info->base_op_run_info.output_tensors[0];
  } else {
    std::vector<ValuePtr> output_values;
    std::transform(op_run_info->base_op_run_info.output_tensors.begin(),
                   op_run_info->base_op_run_info.output_tensors.end(), std::back_inserter(output_values),
                   [](const auto &t) {
                     MS_EXCEPTION_IF_NULL(t);
                     return t;
                   });
    op_run_info->real_out = std::make_shared<ValueTuple>(output_values);
  }

  UpdateOutputStubNodeValue(op_run_info);
}

bool ForwardExecutor::ProcessViewOp(const FrontendOpRunInfoPtr &op_run_info,
                                    const ops::StridesCalcFunc &strides_calc_func, bool is_tuple_output) {
  if (!EnableView(op_run_info)) {
    return false;
  }
  MS_LOG(DEBUG) << "Start, op:" << op_run_info->base_op_run_info.op_name;
  if (op_run_info->op_grad_info->input_value.empty()) {
    MS_LOG(EXCEPTION) << "op_run_info->op_grad_info->input_value is empty";
  }

  // Only split and chunk has mul outputs, and input tensor is first input.
  auto view_value = op_run_info->op_grad_info->input_value[0];
  MS_EXCEPTION_IF_NULL(view_value);
  if (!view_value->isa<tensor::Tensor>()) {
    MS_EXCEPTION(TypeError) << "input value is not Tensor";
  }
  auto view_input_tensor = view_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(view_input_tensor);
  DeviceSyncPtr input_origin_device_address{nullptr};
  std::shared_ptr<tensor::FutureBase<DeviceSync>> input_origin_address_future{nullptr};
  if (view_input_tensor->storage_info() != nullptr) {
    input_origin_address_future = view_input_tensor->address_future();
    if (input_origin_address_future == nullptr) {
      input_origin_device_address = view_input_tensor->device_address();
    }
  }

  auto storage_infos = strides_calc_func(op_run_info->op_grad_info->op_prim, op_run_info->op_grad_info->input_value);
  if (storage_infos.empty()) {
    MS_LOG(DEBUG) << "Not View op " << op_run_info->base_op_run_info.op_name;
    return false;
  }

  op_run_info->is_view_op = true;
  // Reuse SetInputAbstract, abs of inputs is need when requires_grad is true.
  InferOutputAbstract(op_run_info);
  CheckIfNeedSyncForHeterogeneous(op_run_info->base_op_run_info.device_target);

  // Create view output tensor
  CreateViewOpOutputs(op_run_info, view_input_tensor, storage_infos, input_origin_address_future,
                      input_origin_device_address, is_tuple_output);

  KernelTaskType task_type = GetViewOpTaskType(op_run_info->base_op_run_info.op_name);
  if (op_run_info->requires_grad || task_type != KernelTaskType::kNORMAL_VIEW_TASK) {
    const auto &top_cell = op_run_info->requires_grad ? grad()->top_cell() : nullptr;
    for (size_t index = 0; index < op_run_info->input_size; ++index) {
      const ValuePtr &input_object = op_run_info->op_grad_info->input_value[index];
      PyNativeAlgo::DataConvert::ConvertValueToTensor(op_run_info, input_object, index, top_cell);
    }
  }
  if (EnablePipeline(op_run_info->base_op_run_info.op_name)) {
    if (task_type == KernelTaskType::kNORMAL_VIEW_TASK && !op_run_info->requires_grad) {
      MS_LOG(DEBUG) << "End";
      return true;
    }
    DispatchViewKernelTask(op_run_info, task_type);
  } else {
    // Gil might be release  by ACL, so release here to reduce conflict
    GilReleaseWithCheck release_gil;
    backend_queue_->Wait();
    ForwardRunViewKernelTask(op_run_info, task_type, false);
    ForwardOpGradImpl(op_run_info);
  }
  MS_LOG(DEBUG) << "End";
  return true;
}

void ForwardExecutor::DispatchSilceOpFrontendTask(const std::vector<ValuePtr> &input_values,
                                                  const std::vector<SliceOpInfoPtr> &slice_op_infos, bool requires_grad,
                                                  const stub::StubNodePtr &stub_output) {
  auto forward_task = std::make_shared<SliceOpFrontendTask>(
    [this](const std::vector<ValuePtr> &input_values, const std::vector<SliceOpInfoPtr> &slice_op_infos,
           bool requires_grad, const stub::StubNodePtr &stub_output) {
      (void)RunSliceOpFrontend(input_values, slice_op_infos, requires_grad, stub_output);
    },
    input_values, slice_op_infos, requires_grad, stub_output);
  frontend_queue_->Push(forward_task);
}

ValuePtr ForwardExecutor::RunSliceOpFrontend(const std::vector<ValuePtr> &input_values,
                                             const std::vector<SliceOpInfoPtr> &slice_op_infos, bool requires_grad,
                                             const stub::StubNodePtr &stub_output) {
  if (input_values.empty()) {
    MS_LOG(EXCEPTION) << "input_values is empty.";
  }

  MS_LOG(DEBUG) << "Start, slice_op_infos size:" << slice_op_infos.size();
  auto intermediate_tensor = input_values;
  auto last_tensor = input_values[0];

  for (size_t i = 0; i < slice_op_infos.size(); i++) {
    auto slice_op_info = slice_op_infos[i];
    MS_EXCEPTION_IF_NULL(slice_op_info);
    MS_LOG(DEBUG) << "Run slice op name:" << slice_op_info->slice_op_name;
    MS_EXCEPTION_IF_CHECK_FAIL(!slice_op_info->data_indexs.empty(), "data_indexs can not be empty");
    auto first_data_idx = slice_op_info->data_indexs[0];
    if (first_data_idx >= intermediate_tensor.size()) {
      MS_LOG(EXCEPTION) << "data_idx is out of bounds, data_idx:" << first_data_idx
                        << " intermediate_tensor.size():" << intermediate_tensor.size();
    }

    // Only last op need to update stub node.
    auto cur_op_stub_output = (i + 1 == slice_op_infos.size() ? stub_output : nullptr);
    auto op_run_info = GenerateSliceOpRunInfo(slice_op_info->slice_op_name, requires_grad, cur_op_stub_output);
    if (slice_op_info->slice_op_name == kCastOpName) {
      // slice_index_inputs of Cast op is type
      MS_EXCEPTION_IF_CHECK_FAIL(slice_op_info->slice_index_inputs.size() == 1, "Size of cast type input should be 1");
      auto type_value = slice_op_info->slice_index_inputs[0];
      MS_EXCEPTION_IF_CHECK_FAIL(type_value->is_int(), "type_value should be int.");
      auto type_id = static_cast<TypeId>(type_value->int_value());
      cast_operation()->DoNormalCast(op_run_info, intermediate_tensor[first_data_idx], type_id);
    } else {
      EmplaceSliceInputs(op_run_info, intermediate_tensor, slice_op_info);
      PyNativeAlgo::Common::StubNodeToValue(op_run_info);

      auto strides_calc_info =
        ops::ViewStridesCalcFactory::GetInstance().GetStridesCalcFunc(op_run_info->base_op_run_info.op_name);
      if (!strides_calc_info.first.has_value()) {
        MS_LOG(EXCEPTION) << "op:" << op_run_info->base_op_run_info.op_name << " is not view.";
      }
      if (!ProcessViewOp(op_run_info, strides_calc_info.first.value(), strides_calc_info.second)) {
        MS_EXCEPTION(ValueError) << "op:" << op_run_info->base_op_run_info.op_name << " inputs is not for view.";
      }
    }
    intermediate_tensor[first_data_idx] = op_run_info->real_out;
    last_tensor = op_run_info->real_out;
  }
  MS_LOG(DEBUG) << "End";
  return last_tensor;
}

void ForwardExecutor::RunOpFrontend(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "RunOp name: " << op_run_info->base_op_run_info.op_name;
  // Convert StubNode to Tensor and no need to concern about input StubNode anymore in this thread.
  PyNativeAlgo::Common::StubNodeToValue(op_run_info);
  // 1.Set cast for inputs
  SetCastForInputs(op_run_info);

#ifndef ENABLE_TEST
  auto strides_calc_info =
    ops::ViewStridesCalcFactory::GetInstance().GetStridesCalcFunc(op_run_info->base_op_run_info.op_name);
  // Ascend Op not support, We will remove it next week;
  if (strides_calc_info.first.has_value() &&
      ProcessViewOp(op_run_info, strides_calc_info.first.value(), strides_calc_info.second)) {
    return;
  }
#endif
  op_run_info->is_view_op = false;

  // Infer output abstract
  InferOutputAbstract(op_run_info);

  if (!op_run_info->base_op_run_info.has_dynamic_output) {
    // Output is dynamic shape, need to SetAbstract after RunOp.
    UpdateOutputStubNodeAbs(op_run_info);
  }

  if (op_run_info->output_get_by_infer_value) {
    UpdateOutputStubNodeValue(op_run_info);
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
  {
    GilReleaseWithCheck gil_release;
    backend_queue_->Wait();
  }
  const auto &backend_op_run_info = CreateBackendOpRunInfo(op_run_info);
  RunOpBackend(op_run_info, backend_op_run_info);
  if (!op_run_info->requires_grad) {
    MS_LOG(DEBUG) << "Grad flag is false";
    UpdateStubTensor(op_run_info);
    return;
  }
  // 4. Do op grad and record op info
  ForwardOpGradImpl(op_run_info);
  // output is dynamic shape. Need to update abstract and value.
  UpdateStubTensor(op_run_info);
}

void ForwardExecutor::OpRunInfoUsePrimC(const FrontendOpRunInfoPtr &op_run_info) const {
  auto prim = op_run_info->op_grad_info->op_prim;
  auto op_name = prim->name();
  if (EnablePipeline(op_name) && expander::bprop::HasBpropExpander(op_name) &&
      abstract::GetFrontendPrimitiveInferImpl(prim).has_value()) {
    auto new_prim = std::make_shared<Primitive>(*prim);
    new_prim->EnableSharedMutex();
    op_run_info->op_grad_info->op_prim = new_prim;
  }
}

PrimitivePtr ForwardExecutor::GetSlicePrimFromCache(const std::string &op_name, bool is_input_to_attr) {
  auto iter = slice_prim_cache_.find(op_name);
  if (iter != slice_prim_cache_.end()) {
    if (is_input_to_attr) {
      return std::make_shared<Primitive>(*iter->second);
    }
    return iter->second;
  }

  auto prim = std::make_shared<Primitive>(op_name);
  if (op_name == kStridedSliceOpName) {
    int64_t v = 0;
    prim->set_attr(kAttrBeginMask, MakeValue(v));
    prim->set_attr(kAttrEndMask, MakeValue(v));
    prim->set_attr(kAttrEllipsisMask, MakeValue(v));
    prim->set_attr(kAttrNewAxisMask, MakeValue(v));
    prim->set_attr(kAttrShrinkAxisMask, MakeValue(v));
  }
  slice_prim_cache_[op_name] = prim;
  return prim;
}

FrontendOpRunInfoPtr ForwardExecutor::GenerateSliceOpRunInfo(const std::string &op_name, bool requires_grad,
                                                             const stub::StubNodePtr &stub_output) {
  Init();
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->base_op_run_info.op_name = op_name;
  op_run_info->requires_grad = requires_grad;
  op_run_info->base_op_run_info.device_target = device_target_;

  if (op_name == kCastOpName) {
    // Cast prim will be set in DoNormalCast.
    return op_run_info;
  }

  bool is_input_to_attr = kSliceOpInputToAttr.find(op_name) != kSliceOpInputToAttr.end();
  if (op_run_info->requires_grad || is_input_to_attr) {
    op_run_info->op_grad_info->op_prim = GetSlicePrimFromCache(op_name, is_input_to_attr);
  }
  op_run_info->stub_output = stub_output;
  return op_run_info;
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
    op_run_info->base_op_run_info.use_dynamic_shape_process =
      grad()->forward_use_dynamic_shape_process() || grad()->use_dynamic_shape_process();
  }
  PyNativeAlgo::PyParser::SetPrim(op_run_info, args[static_cast<size_t>(RunOpArgsEnum::PY_PRIM)]);
  OpRunInfoUsePrimC(op_run_info);
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
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, true);

  VectorRef outputs;
  const auto &cur_mind_rt_backend = GetMindRtBackend(backend_op_run_info->base_op_run_info.device_target);
  MS_EXCEPTION_IF_NULL(cur_mind_rt_backend);
  bool use_dynamic_shape_process = backend_op_run_info->base_op_run_info.use_dynamic_shape_process;
  if (use_dynamic_shape_process) {
    cur_mind_rt_backend->RunOpDynamic(backend_op_run_info, &outputs);
  } else {
    cur_mind_rt_backend->RunOp(backend_op_run_info, &outputs);
  }

  if (op_run_info->base_op_run_info.has_dynamic_output ||
      OpCompiler::GetInstance().IsInvalidInferResultOp(op_run_info->base_op_run_info.op_name)) {
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
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto backend = std::make_shared<compile::MindRTBackend>("ms", cur_device_target, device_id);
    MS_EXCEPTION_IF_NULL(backend);
    mindrt_backends_[cur_device_target] = backend;
    return backend;
  }
}

ValuePtr ForwardExecutor::RunOpWithBackendPolicy(const FrontendOpRunInfoPtr &op_run_info,
                                                 const BackendOpRunInfoPtr &backend_op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
#ifndef ENABLE_TEST
  if (IsVmOp(op_run_info->base_op_run_info.op_name)) {
    return RunOpInVM(op_run_info);
  } else {
    return RunOpInMs(op_run_info, backend_op_run_info);
  }
#else
  return RunOpInVM(op_run_info);
#endif
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

void ForwardExecutor::ProcessBeforeNewGraph(const py::object &obj, const py::args &args) {
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
  grad()->dynamic_shape()->UpdateArgsAbsToUnknownShapeAbs(obj, args);
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
#if defined(__APPLE__)
    ClearNodeAbsMap();
#else
    auto forward_task = std::make_shared<FrontendTask>([this](...) { ClearNodeAbsMap(); }, nullptr);
    frontend_queue_->Push(forward_task);
#endif
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

void ForwardExecutor::CreateInputAddressForViewOp(const tensor::TensorPtr &input_tensor,
                                                  const FrontendOpRunInfoPtr &op_run_info, const size_t &input_idx) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto check_view_device_address = [&input_tensor]() {
    if (input_tensor->address_future() == nullptr && input_tensor->device_address() != nullptr) {
      input_tensor->device_address()->set_is_view(true);
      return true;
    }
    return false;
  };
  if (check_view_device_address()) {
    return;
  }
  if (input_tensor->address_future() == nullptr) {
    runtime::OpExecutor::GetInstance().WaitAll();
  }
  if (check_view_device_address()) {
    return;
  }

  MS_LOG(DEBUG) << "Input_tensor address is nullptr, need create address.";
  if (EnablePipeline(op_run_info->base_op_run_info.op_name)) {
    if (input_tensor->address_future() != nullptr) {
      DispatchAllocateMemTask(op_run_info, input_tensor, input_idx, true);
    } else {
      DeviceAddressPromisePtr promise =
        std::make_unique<DeviceAddressPromise>(std::promise<DeviceAddressFutureDataPtr>());
      auto future = promise->GetFuture();
      auto device_address_future = std::make_shared<DeviceAddressFuture>(std::move(future));
      input_tensor->set_address_future(device_address_future);
      (void)op_run_info->device_sync_promises.emplace_back(std::move(promise));
      DispatchAllocateMemTask(op_run_info, input_tensor, input_idx, false);
    }
  } else {
    // Sync address_future is nullptr
    CreateDeviceAddressForViewInput(op_run_info, input_tensor, input_idx, false);
  }
}

void ForwardExecutor::DispatchAllocateMemTask(const FrontendOpRunInfoPtr &op_run_info,
                                              const tensor::TensorPtr &input_tensor, const size_t &input_idx,
                                              bool need_wait) {
  static auto alloc_mem_func = [this](const FrontendOpRunInfoPtr &op_run_info, const tensor::TensorPtr &input_tensor,
                                      const size_t &input_idx, bool need_wait) {
    CreateDeviceAddressForViewInput(op_run_info, input_tensor, input_idx, true, need_wait);
  };

  auto view_task =
    std::make_shared<AllocViewMemBackendTask>(alloc_mem_func, op_run_info, input_tensor, input_idx, need_wait);
  backend_queue_->Push(view_task);
}

void ForwardExecutor::RefreshTensorContiguous(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->storage_info() == nullptr) {
    return;
  }

  // Gil might be release  by ACL, so release here to reduce conflict
  GilReleaseWithCheck release_gil;
  if (!ScopedFallbackRunning::on() && enable_async()) {
    static auto contiguous_func = [this](const tensor::TensorPtr &tensor) { RunContiguousTask(tensor, true); };

    auto contiguous_task = std::make_shared<ContiguousBackendTask>(contiguous_func, tensor);
    backend_queue_->Push(contiguous_task);
  } else {
    backend_queue_->Wait();
    RunContiguousTask(tensor, false);
  }
}

void ForwardExecutor::RunContiguousTaskForTensor(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->storage_info() == nullptr) {
    return;
  }

  GilReleaseWithCheck release_gil;
  RunContiguousTask(tensor, !ScopedFallbackRunning::on() && enable_async());
}

device::DeviceAddressPtr ForwardExecutor::TensorContiguousCallback(const DeviceSyncPtr &device_address,
                                                                   const TensorStorageInfoPtr &storage_info) {
  MS_EXCEPTION_IF_NULL(device_address);
  // Gil might be release  by ACL, so release here to reduce conflict
  auto device_addr = std::dynamic_pointer_cast<device::DeviceAddress>(device_address);
  MS_EXCEPTION_IF_NULL(device_addr);
  if (storage_info == nullptr) {
    return device_addr;
  }

  GilReleaseWithCheck release_gil;
  const auto &cur_mind_rt_backend = GetMindRtBackend(device_addr->device_name());
  MS_EXCEPTION_IF_NULL(cur_mind_rt_backend);
  // as_numpy sync promise contiguous run_sync
  auto ret = cur_mind_rt_backend->RunContiguousTaskByAddress(device_addr, storage_info, false);
  runtime::OpExecutor::GetInstance().WaitAll();
  return ret;
}

void ForwardExecutor::PrepareOpInputs(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  CheckIfNeedSyncForHeterogeneous(op_run_info->base_op_run_info.device_target);
  PyNativeAlgo::DataConvert::GetInputTensor(op_run_info, op_run_info->requires_grad ? grad()->top_cell() : nullptr);
  for (const auto &tensor : op_run_info->base_op_run_info.input_tensor) {
    RefreshTensorContiguous(tensor);
  }
}

void ForwardExecutor::CreateViewOutputTensor(
  const FrontendOpRunInfoPtr &op_run_info, const tensor::TensorPtr &input_tensor,
  const TensorStorageInfoPtr &storage_info,
  const std::shared_ptr<tensor::FutureBase<DeviceSync>> &input_origin_address_future,
  const DeviceSyncPtr &input_origin_device_address, const TypePtr &real_type) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(storage_info);
  auto output_tensor = std::make_shared<tensor::Tensor>(real_type->type_id(), storage_info->shape);
  output_tensor->set_storage_info(storage_info);
  output_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
  output_tensor->set_contiguous_callback([this](const tensor::TensorPtr &tensor, const DeviceSyncPtr &device_address,
                                                const TensorStorageInfoPtr &storage_info) -> DeviceSyncPtr {
    if (tensor != nullptr) {
      frontend_queue_->Wait();
      backend_queue_->Wait();

      auto new_addr = TensorContiguousCallback(tensor->device_address(), tensor->storage_info());
      tensor->set_device_address(new_addr);
      tensor->set_storage_info(nullptr);

      return nullptr;
    }
    return TensorContiguousCallback(device_address, storage_info);
  });

  auto address_future =
    input_origin_address_future == nullptr ? input_tensor->address_future() : input_origin_address_future;
  if (address_future != nullptr) {
    output_tensor->set_address_future(address_future);
  } else {
    auto device_address =
      input_origin_device_address == nullptr ? input_tensor->device_address() : input_origin_device_address;
    output_tensor->set_device_address(device_address);
  }
  if (op_run_info->requires_grad) {
    output_tensor->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>());
    output_tensor->auto_grad_meta_data()->set_grad_type(TensorGradType::kOpOutput);
  }
  (void)op_run_info->base_op_run_info.output_tensors.emplace_back(output_tensor);
}

void ForwardExecutor::PrepareOpOutputs(const FrontendOpRunInfoPtr &op_run_info) const {
  CreateOutputTensor(op_run_info->base_op_run_info.abstract, &op_run_info->base_op_run_info.output_tensors,
                     &op_run_info->device_sync_promises);
  TransformOutputValues(op_run_info);
  UpdateOutputStubNodeValue(op_run_info);
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
  is_jit_compiling_ = false;
  cast_operation()->ClearRes();
  ClearNodeAbsMap();
  infer_operation()->ClearPrimAbsList();
  infer_operation()->ClearConstFlagPrimCache();
  std::stack<CellPtr>().swap(forward_cell_stack_);
  mindrt_backends_.clear();
  slice_prim_cache_.clear();
}
}  // namespace pynative
}  // namespace mindspore
