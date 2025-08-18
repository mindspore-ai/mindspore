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

#include "pynative/forward/op_graph/forward.h"
#include <set>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <utility>
#include <string>
#include <stack>
#include <memory>
#include "ir/tensor_new.h"
#include "primitive/structure_op_name.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "include/pynative/utils/pynative_utils.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/utils/amp.h"
#include "include/utils/pipeline/python_fallback_running.h"
#include "utils/ms_context.h"
#include "include/pynative/forward/pyboost/forward_task.h"
#include "pynative/utils/predict_out_type_map.h"
#include "include/utils/stub_tensor.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "tools/profiler/profiling.h"
using mindspore::profiler::ProfilerManager;
#include "include/frontend/expander/bprop_interface.h"
#include "include/frontend/jit/ps/parse/py_data_convert.h"
#include "include/frontend/operator/frontend_primitive_infer.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/backend/common/device_address_utils.h"
#include "include/backend/common/kernel_graph/session_basic.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "pynative/backward/grad_utils.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_reg.h"
#include "include/pynative/utils/pyboost/auto_generate/contiguous.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "include/utils/tensor_py.h"
#include "utils/stream_guard.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/runtime/utils/dispatch/dispatch_env.h"
#include "include/pynative/utils/pyboost/functions/dispatch.h"

namespace mindspore {
namespace pynative {
enum class RunOpArgsEnum : size_t { PY_PRIM = 0, PY_NAME, PY_INPUTS, PY_ARGS_NUM };
namespace {
const std::set<std::string> kVmOperators = {"InsertGradientOf", "StopGradient", "HookBackward", "CellBackwardHook"};
constexpr char kBegin[] = "Begin";
constexpr char kEnd[] = "End";
constexpr auto kOpNameCustom = "Custom";

#ifndef ENABLE_TEST
void CreateDeviceAddressForTensor(const FrontendOpRunInfoPtr &op_run_info, const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->device_address() != nullptr) {
    return;
  }
  // Create a device address for tensor
  const auto &device_context = runtime::OpRunner::GetDeviceContext(op_run_info->base_op_run_info.device_target);
  MS_EXCEPTION_IF_NULL(device_context);
  auto tensor_size = LongToSize(tensor->DataNBytes());
  runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, op_run_info->base_op_run_info.stream_id,
                                                         tensor, tensor_size);
  // Allocate a block of device memory
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kWaitTaskFinish,
                                     runtime::kDefaultOpName);
  GilReleaseWithCheck gil_release;
  runtime::Pipeline::Get().backend_stage()->Wait();
  runtime::Pipeline::Get().launch_stage()->Wait();

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", op_run_info->base_op_run_info.op_name, "");
  kernel::pyboost::PyBoostUtils::MallocForInput(device_context, tensor);
  static bool enable_tracker = device::tracker::MemTrackerManager::GetInstance().IsEnabled();
  if (MS_UNLIKELY(enable_tracker)) {
    auto device_address = tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
      MarkTensorAsInput, "PyNative", device::GetDeviceNameByType(device_address->GetDeviceType()),
      device_address->GetPtr(), tensor->data_type(), tensor->shape(), tensor->storage_info());
  }
}
#endif

ValuePtr CopyTensorValueWithNewId(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
#ifndef ENABLE_TEST
    // Preallocate device memory for the tensor to prevent the device memory of the same tensor
    // from being allocated multiple times
    CreateDeviceAddressForTensor(op_run_info, tensor);
#endif
    // This constructor will make a tensor with the new id
    auto new_tensor = tensor::from_spec(tensor->data_type(), tensor->shape(), device::DeviceType::kNone);
    // todo: check tensor->data need ?
    new_tensor->set_need_pipeline_sync(true);
    new_tensor->set_device_address(tensor->device_address());
    new_tensor->set_format(tensor->format());
    new_tensor->set_storage_info(tensor->storage_info());
    new_tensor->set_contiguous_callback(tensor->contiguous_callback());
    new_tensor->set_sync_status(tensor->sync_status());
    return new_tensor;
  }
  if (v->isa<ValueTuple>()) {
    const auto &v_tup = v->cast<ValueTuplePtr>();
    ValuePtrList list;
    for (const auto &ele : v_tup->value()) {
      (void)list.emplace_back(CopyTensorValueWithNewId(op_run_info, ele));
    }
    return std::make_shared<ValueTuple>(list);
  }
  if (v->isa<ValueList>()) {
    const auto &v_list = v->cast<ValueListPtr>();
    ValuePtrList list;
    for (const auto &ele : v_list->value()) {
      (void)list.emplace_back(CopyTensorValueWithNewId(op_run_info, ele));
    }
    return std::make_shared<ValueList>(list);
  }
  if (v->isa<ValueDictionary>()) {
    const auto &v_dict = v->cast<ValueDictionaryPtr>();
    std::vector<std::pair<ValuePtr, ValuePtr>> res;
    res.reserve(v_dict->value().size());
    for (const auto &[k, v] : v_dict->value()) {
      (void)res.emplace_back(std::make_pair(k, CopyTensorValueWithNewId(op_run_info, v)));
    }
    return std::make_shared<ValueDictionary>(res);
  }
  return v;
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

ValuePtr ConstructOutputInVM(const std::vector<ValuePtr> &result) {
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
  auto backend_op_run_info = std::make_shared<session::BackendOpRunInfo>(
    op_run_info->base_op_run_info, std::make_shared<Primitive>(*op_run_info->op_grad_info->op_prim), true, false);
  // Erase RandomOp cache avoid memory leak.
  if (AnfAlgo::NeedEraseCache(backend_op_run_info->op_prim)) {
    backend_op_run_info->base_op_run_info.need_earse_cache = true;
  }
  if (op_run_info->base_op_run_info.has_dynamic_output) {
    backend_op_run_info->base_op_run_info.use_dynamic_shape_process = true;
  }
  return backend_op_run_info;
}

runtime::KernelTaskType GetViewOpTaskType(const std::string &op_name) {
  if (op_name == kCopyWithSliceOpName) {
    return runtime::KernelTaskType::kCOPY_TASK;
  }
  return runtime::KernelTaskType::kNORMAL_VIEW_TASK;
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

  for (const auto &slice_index : slice_op_info->slice_index_inputs) {
    ValuePtr v = nullptr;
    if (slice_index->is_int()) {
      v = MakeValue(slice_index->int_value());
    } else {
      v = MakeValue(slice_index->vec_value());
    }

    (void)op_run_info->op_grad_info->input_value.emplace_back(v);
  }

  if (op_run_info->requires_grad && op_run_info->base_op_run_info.op_name == kStridedSliceOpName) {
    // StridedSlice mask input
    int64_t v = 0;

    (void)op_run_info->op_grad_info->input_value.emplace_back(MakeValue(v));  // begin_mask
    (void)op_run_info->op_grad_info->input_value.emplace_back(MakeValue(v));  // end_mask
    (void)op_run_info->op_grad_info->input_value.emplace_back(MakeValue(v));  // ellipsis_mask
    (void)op_run_info->op_grad_info->input_value.emplace_back(MakeValue(v));  // new_axis_mask
    (void)op_run_info->op_grad_info->input_value.emplace_back(MakeValue(v));  // shrink_new_mask
  }

  op_run_info->input_size = op_run_info->op_grad_info->input_value.size();
  op_run_info->op_grad_info->input_value_grad_type.resize(op_run_info->input_size);
}

bool GetMixprecisionTypeFromStrategy(const PyboostOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto cur_amp_Strategy = amp::GetCurrentAmpStrategy();
  if (cur_amp_Strategy == nullptr || cur_amp_Strategy->GetAmpLevel() == amp::AmpLevel::O0) {
    return false;
  }
  const auto &op_cast_strategy_info = GetPrimCastStrategyInfo(cur_amp_Strategy, op_run_info->op_prim->name());
  if (op_cast_strategy_info.strategy == amp::Ignore) {
    return false;
  }
  if (op_cast_strategy_info.strategy == amp::AutoPromote) {
    op_run_info->mix_type = kAutoPromote;
  }
  op_run_info->mix_precision_type = op_cast_strategy_info.dtype;
  return true;
}

bool GetMixprecisionTypeFromStrategy(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto cur_amp_Strategy = amp::GetCurrentAmpStrategy();
  if (cur_amp_Strategy == nullptr || cur_amp_Strategy->GetAmpLevel() == amp::AmpLevel::O0) {
    return false;
  }
  const auto &op_cast_strategy_info =
    GetPrimCastStrategyInfo(cur_amp_Strategy, op_run_info->op_grad_info->op_prim->name());
  if (op_cast_strategy_info.strategy == amp::Ignore) {
    return false;
  }
  if (op_cast_strategy_info.strategy == amp::AutoPromote) {
    op_run_info->mix_type = kAutoPromote;
  }
  op_run_info->mix_precision_type = op_cast_strategy_info.dtype;
  return true;
}

tensor::TensorPtr TensorContiguous(const tensor::TensorPtr &tensor) {
  if (tensor == nullptr || tensor->storage_info() == nullptr) {
    return tensor;
  }
  auto old_device_address = tensor->device_address();
  MS_EXCEPTION_IF_NULL(old_device_address);

  const DeviceContext *device_context = runtime::OpRunner::GetDeviceContext(old_device_address->GetDeviceType());
  MS_EXCEPTION_IF_NULL(device_context);
  GilReleaseWithCheck release_gil;
  auto contiguous_op = CREATE_PYBOOST_OP(Contiguous, device_context->GetDeviceType());
  const auto &contiguous_tensor = contiguous_op->Call(tensor);
  // Need to wait contiguous finish in heterogeneous scenarios.
  runtime::Pipeline::Get().WaitForward();
  return contiguous_tensor;
}

void ContiguousInputByRunInfo(const BackendOpRunInfoPtr &op_run_info) {
  auto &inputs = op_run_info->base_op_run_info.expanded_input_values;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    if (!input->isa<tensor::Tensor>()) {
      continue;
    }
    auto tensor = input->cast<tensor::TensorPtr>();
    if (tensor->storage_info() == nullptr) {
      continue;
    }

    auto contiguous_tensor = TensorContiguous(tensor);
    inputs[i] = contiguous_tensor;
  }
}

bool HasBpropExpander(const std::string &prim_name) {
  auto handle = expander::bprop::GetBpropIRBuilder(prim_name);
  return handle != nullptr;
}

void UpdateDeviceByInputs(const FrontendOpRunInfoPtr &op_run_info) {
  if (EnableDispatch()) {
    kernel::pyboost::OpRunStatus::Get().set_run_info(kernel::pyboost::OpStatus(
      op_run_info->async_status.disable_mix_precision, op_run_info->base_op_run_info.device_target));

    auto device = get_device(op_run_info->op_grad_info->input_value);
    op_run_info->base_op_run_info.device_target = device;
  }
}

void CheckViewInplaceInput(const FrontendOpRunInfoPtr &op_run_info) {
  auto op_def = mindspore::ops::GetOpDef(op_run_info->base_op_run_info.op_name);
  if (op_def == nullptr) {
    return;
  }

  const auto &outputs = op_def->returns_;
  for (const auto &output : outputs) {
    if (output.inplace_input_index_ < 0) {
      continue;
    }

    if (static_cast<size_t>(output.inplace_input_index_) >= op_run_info->op_grad_info->input_value.size()) {
      MS_LOG(EXCEPTION) << "Index " << output.inplace_input_index_ << " out of bound "
                        << op_run_info->op_grad_info->input_value.size();
    }
    const auto &input = op_run_info->op_grad_info->input_value[output.inplace_input_index_];
    MS_EXCEPTION_IF_NULL(input);
    if (input->isa<Tensor>()) {
      auto tensor = input->cast<TensorPtr>();
      if (tensor->storage_info() != nullptr) {
        MS_LOG(EXCEPTION) << "Check view and inplace failed! Not support view input for inplace op " << op_def->name_
                          << ". Got input with storage_info:" << tensor->storage_info()->ToString();
      }
    } else {
      MS_LOG(EXCEPTION) << "The op " << op_def->name_ << " inplace input is not a Tensor!";
    }
  }
}
}  // namespace

void ForwardExecutor::WaitForwardTask() {
  GilReleaseWithCheck gil_release;
  runtime::Pipeline::Get().frontend_stage()->Wait();
}

GradExecutorPtr ForwardExecutor::grad() const {
  auto grad_executor = grad_executor_.lock();
  MS_EXCEPTION_IF_NULL(grad_executor);
  return grad_executor;
}

void ForwardExecutor::InitOpRunInfo(const PyboostOpRunInfoPtr &op_run_info) {
  Init();
  // Used for async run
  op_run_info->requires_grad = GradState::Get().RequiresGrad();
  op_run_info->device_target = GetCurrentDeviceTarget(op_run_info->op_prim);
  // device_res_manager_->GetCurrentStreamId is not correct.
  // The stream_id is always 0 on CPU.
  // Heterogeneous scenarios may have accuracy issues.
  op_run_info->stream_id = CurrentStream::id();
}

void ForwardExecutor::InitOpRunInfo(const FrontendOpRunInfoPtr &op_run_info) {
  Init();
  // Used for async run
  op_run_info->requires_grad = GradState::Get().RequiresGrad();
  op_run_info->base_op_run_info.use_dynamic_shape_process = grad()->forward_use_dynamic_shape_process();
  op_run_info->base_op_run_info.device_target = GetCurrentDeviceTarget(op_run_info->op_grad_info->op_prim);
  op_run_info->base_op_run_info.stream_id = CurrentStream::id();
}

void ForwardExecutor::ReInit() {
  device_target_ = device::GetDeviceTypeByName(MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET));
  bool sync_stream = runtime::RuntimeConf::GetInstance()->launch_blocking();
  enable_async_ = !sync_stream;
}

void ForwardExecutor::Init() {
  ReInit();
  if (init_) {
    return;
  }
  init_ = true;
  MS_LOG(DEBUG) << "Init ForwardExecutor";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_LOG(DEBUG) << "Enable mindRT.";
  context_ptr->set_param<bool>(MS_CTX_ENABLE_MINDRT, true);
  python_adapter::set_python_env_flag(true);
  tensor::Tensor::RegisterLazyCallback([]() { runtime::Pipeline::Get().WaitAll(); });
  runtime::OpExecutor::GetInstance().RegisterCallbackForMemoryPool();
  op_backend_ = std::make_unique<compile::OpBackend>();
  compile::ViewBackend::SetContiguousFunc(
    [](const BackendOpRunInfoPtr &op_run_info) { ContiguousInputByRunInfo(op_run_info); });
}

void ForwardExecutor::RefreshForwardCallback() {
#if defined(_WIN32) || defined(_WIN64)
  tensor::Tensor::RegisterLazyCallback([]() { runtime::Pipeline::Get().WaitAll(); });
#endif
  // ForwardCallback has been set in ForwardExecutor::Init, no need to refresh anymore.
}

bool ForwardExecutor::enable_async() const {
#if defined(ENABLE_TEST) || defined(__APPLE__)
  return false;
#else
  return enable_async_;
#endif
}

bool ForwardExecutor::EnablePipeline(const std::string &op_name) const {
  return enable_async() && !PyNativeAlgo::Common::IsVmOp(op_name) && op_name != kOpNameCustom &&
         !ScopedFallbackRunning::on() && !runtime::OpExecutor::NeedSync();
}

void ForwardExecutor::DispatchFrontendTask(const FrontendOpRunInfoPtr &op_run_info) {
  auto forward_task = std::make_shared<FrontendTask>(
    [this](const FrontendOpRunInfoPtr &op_run_info) { RunOpFrontend(op_run_info); }, op_run_info);
  runtime::ProfilerAnalyzer::GetInstance().RecordFlowData(forward_task->task_id());
  runtime::Pipeline::Get().frontend_stage()->Push(forward_task);
}

void ForwardExecutor::ForwardOpGradImpl(const OpGradInfoPtr &grad_info, const AsyncStatus &async_status) const {
  grad()->ProcessOpGradInfo(grad_info);
}

void ForwardExecutor::ForwardRunViewKernelTask(const FrontendOpRunInfoPtr &op_run_info,
                                               const runtime::KernelTaskType &task_type, bool enable_async) {
  if (task_type == runtime::KernelTaskType::kNORMAL_VIEW_TASK) {
    return;
  }
  MS_LOG(DEBUG) << "Start, task_type:" << task_type;
  MS_EXCEPTION_IF_NULL(op_backend_);
  op_backend_->RunViewKernelTask(op_run_info->base_op_run_info, task_type, enable_async);

  MS_LOG(DEBUG) << "End";
}

void ForwardExecutor::CreateViewOpOutputs(const FrontendOpRunInfoPtr &op_run_info,
                                          const tensor::TensorPtr &view_input_tensor, runtime::KernelTaskType task_type,
                                          const TensorStorageInfoPtrList &storage_infos, bool is_tuple_output) {
  const bool is_single_tensor_output = storage_infos.size() == 1 && !is_tuple_output;
  // Generate output abs by storage_info.
  if (is_single_tensor_output) {
    op_run_info->base_op_run_info.abstract = abstract::MakeAbstractTensor(
      std::make_shared<abstract::Shape>(storage_infos[0]->shape), view_input_tensor->Dtype());
  } else {
    AbstractBasePtrList abs_list;
    for (const auto &storage_info : storage_infos) {
      auto abs = abstract::MakeAbstractTensor(std::make_shared<abstract::Shape>(storage_info->shape),
                                              view_input_tensor->Dtype());
      (void)abs_list.emplace_back(abs);
    }
    op_run_info->base_op_run_info.abstract = std::make_shared<abstract::AbstractTuple>(abs_list);
  }

  UpdateOutputStubNodeAbs(op_run_info);
  CreateInputAddressForViewOp(view_input_tensor, op_run_info);

  for (size_t i = 0; i < storage_infos.size(); i++) {
    MS_LOG(DEBUG) << "View op " << op_run_info->base_op_run_info.op_name << ", i:" << i
                  << ", storage_info:" << storage_infos[i]->ToString();
    CreateViewOutputTensor(op_run_info, view_input_tensor, storage_infos[i], task_type, !is_single_tensor_output);
  }

  if (is_single_tensor_output) {
    op_run_info->real_out = op_run_info->base_op_run_info.output_tensors[0];
  } else {
    std::vector<ValuePtr> output_values;
    (void)std::transform(op_run_info->base_op_run_info.output_tensors.begin(),
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
  MS_LOG(DEBUG) << "Start, op:" << op_run_info->base_op_run_info.op_name;
  if (op_run_info->op_grad_info->input_value.empty()) {
    MS_LOG(EXCEPTION) << "op_run_info->op_grad_info->input_value is empty";
  }

  // Only split and chunk has mul outputs, and input tensor is first input.
  auto view_value = op_run_info->op_grad_info->input_value[0];
  MS_EXCEPTION_IF_NULL(view_value);
  if (!view_value->isa<tensor::Tensor>()) {
    MS_EXCEPTION(TypeError) << "For primitive[" << op_run_info->base_op_run_info.op_name
                            << "],  the input[0] should be Tensor, but got:" << view_value->ToString();
  }
  auto view_input_tensor = view_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(view_input_tensor);

  auto storage_infos = strides_calc_func(op_run_info->op_grad_info->op_prim, op_run_info->op_grad_info->input_value);
  if (storage_infos.empty()) {
    MS_LOG(DEBUG) << "Not View op " << op_run_info->base_op_run_info.op_name;
    return false;
  }

  // Reuse SetInputAbstract, abs of inputs is need when requires_grad is true.
  InferOutputAbstract(op_run_info);
  runtime::KernelTaskType task_type = GetViewOpTaskType(op_run_info->base_op_run_info.op_name);

  // Create view output tensor
  CreateViewOpOutputs(op_run_info, view_input_tensor, task_type, storage_infos, is_tuple_output);

  if (op_run_info->requires_grad || task_type != runtime::KernelTaskType::kNORMAL_VIEW_TASK) {
    for (size_t index = 0; index < op_run_info->input_size; ++index) {
      const ValuePtr &input_object = op_run_info->op_grad_info->input_value[index];
      PyNativeAlgo::DataConvert::MarkInputs(op_run_info, input_object, index);
    }
  }

  // Gil might be release  by ACL, so release here to reduce conflict
  GilReleaseWithCheck release_gil;
  ForwardRunViewKernelTask(op_run_info, task_type, false);
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->out_value = op_run_info->real_out;
    op_run_info->op_grad_info->out_abs = op_run_info->base_op_run_info.abstract;
    ForwardOpGradImpl(op_run_info->op_grad_info, op_run_info->async_status);
  }
  MS_LOG(DEBUG) << "End";
  return true;
}

void ForwardExecutor::DispatchSilceOpFrontendTask(const std::vector<ValuePtr> &input_values,
                                                  const std::vector<SliceOpInfoPtr> &slice_op_infos, bool requires_grad,
                                                  const stub::StubNodePtr &stub_output, size_t stream_id) {
  auto forward_task = std::make_shared<SliceOpFrontendTask>(
    [this](const std::vector<ValuePtr> &input_values, const std::vector<SliceOpInfoPtr> &slice_op_infos,
           bool requires_grad, const stub::StubNodePtr &stub_output, size_t stream_id) {
      (void)RunSliceOpFrontend(input_values, slice_op_infos, requires_grad, stub_output, stream_id);
    },
    input_values, slice_op_infos, requires_grad, stub_output, stream_id);
  runtime::Pipeline::Get().frontend_stage()->Push(forward_task);
}

ValuePtr ForwardExecutor::RunSliceOpFrontend(const std::vector<ValuePtr> &input_values,
                                             const std::vector<SliceOpInfoPtr> &slice_op_infos, bool requires_grad,
                                             const stub::StubNodePtr &stub_output, size_t stream_id) {
  if (input_values.empty()) {
    MS_LOG(EXCEPTION) << "input_values is empty.";
  }

  MS_LOG(DEBUG) << "Start, slice_op_infos size:" << slice_op_infos.size();
  auto intermediate_tensor = input_values;
  auto last_tensor = input_values[0];

  for (size_t i = 0; i < slice_op_infos.size(); i++) {
    const auto &slice_op_info = slice_op_infos[i];
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
    auto op_run_info =
      GenerateSliceOpRunInfo(slice_op_info->slice_op_name, requires_grad, cur_op_stub_output, stream_id);
    kernel::pyboost::OpRunStatus::Get().HeterBarrier(op_run_info->base_op_run_info.device_target);
    if (slice_op_info->slice_op_name == kCastOpName) {
      // slice_index_inputs of Cast op is type
      MS_EXCEPTION_IF_CHECK_FAIL(slice_op_info->slice_index_inputs.size() == 1, "Size of cast type input should be 1");
      auto type_value = slice_op_info->slice_index_inputs[0];
      MS_EXCEPTION_IF_CHECK_FAIL(type_value->is_int(), "type_value should be int.");
      auto type_id = static_cast<TypeId>(type_value->int_value());
      (void)cast_operation()->DoNormalCast(op_run_info, intermediate_tensor[first_data_idx], type_id);
    } else {
      EmplaceSliceInputs(op_run_info, intermediate_tensor, slice_op_info);

      auto strides_calc_info =
        ops::ViewStridesCalcFactory::GetInstance().GetStridesCalcFunc(op_run_info->base_op_run_info.op_name);
      if (!strides_calc_info.first.has_value()) {
        MS_LOG(EXCEPTION) << "op:" << op_run_info->base_op_run_info.op_name << " is not view.";
      }
      op_run_info->op_grad_info->operator_type = OperatorType::kViewOp;
      PyNativeAlgo::Common::StubNodeToValue(op_run_info);
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
  UpdateDeviceByInputs(op_run_info);
  kernel::pyboost::OpRunStatus::Get().HeterBarrier(op_run_info->base_op_run_info.device_target);
#ifndef ENABLE_TEST
  auto strides_calc_info =
    ops::ViewStridesCalcFactory::GetInstance().GetStridesCalcFunc(op_run_info->base_op_run_info.op_name);
  op_run_info->op_grad_info->operator_type =
    strides_calc_info.first.has_value() ? OperatorType::kViewOp : OperatorType::kDefault;
#endif

  CheckViewInplaceInput(op_run_info);

  // 1.Set cast for inputs
  SetCastForInputs(op_run_info);

#ifndef ENABLE_TEST
  if (op_run_info->op_grad_info->operator_type == OperatorType::kViewOp &&
      ProcessViewOp(op_run_info, strides_calc_info.first.value(), strides_calc_info.second)) {
    return;
  }
#endif

  op_run_info->op_grad_info->operator_type = OperatorType::kDefault;

  // Infer output abstract
  InferOutputAbstract(op_run_info);

  if (!(op_run_info->base_op_run_info.has_dynamic_output ||
        OpCompiler::GetInstance().IsInvalidInferResultOp(op_run_info->base_op_run_info.op_name))) {
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

  RunOpBackendSync(op_run_info);
}

void ForwardExecutor::RunOpBackendSync(const FrontendOpRunInfoPtr &op_run_info) {
  const auto &backend_op_run_info = CreateBackendOpRunInfo(op_run_info);
  RunOpBackend(op_run_info, backend_op_run_info);
  if (!op_run_info->requires_grad) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }

  op_run_info->op_grad_info->out_value = op_run_info->real_out;
  op_run_info->op_grad_info->out_abs = op_run_info->base_op_run_info.abstract;
  // Do op grad and record op info
  ForwardOpGradImpl(op_run_info->op_grad_info, op_run_info->async_status);
}

void ForwardExecutor::OpRunInfoUsePrimC(const FrontendOpRunInfoPtr &op_run_info) const {
  auto prim = op_run_info->op_grad_info->op_prim;
  auto op_name = prim->name();
  if (EnablePipeline(op_name) && HasBpropExpander(op_name) &&
      abstract::GetFrontendPrimitiveInferImpl(prim).has_value()) {
    auto new_prim = std::make_shared<Primitive>(*prim);
    new_prim->EnableSharedMutex();
    op_run_info->op_grad_info->op_prim = new_prim;
  }
}

PrimitivePtr ForwardExecutor::GetSlicePrimFromCache(const std::string &op_name) {
  auto iter = slice_prim_cache_.find(op_name);
  if (iter != slice_prim_cache_.end()) {
    return iter->second;
  }

  auto prim = std::make_shared<Primitive>(op_name);
  slice_prim_cache_[op_name] = prim;
  return prim;
}

FrontendOpRunInfoPtr ForwardExecutor::GenerateSliceOpRunInfo(const std::string &op_name, bool requires_grad,
                                                             const stub::StubNodePtr &stub_output, size_t stream_id) {
  Init();
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->base_op_run_info.op_name = op_name;
  op_run_info->requires_grad = requires_grad;
  op_run_info->base_op_run_info.device_target = DeviceManagerConf::GetInstance()->device_type();

  op_run_info->base_op_run_info.stream_id = stream_id;

  if (op_name == kCastOpName) {
    // Cast prim will be set in DoNormalCast.
    return op_run_info;
  }

  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->op_prim = GetSlicePrimFromCache(op_name);
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
  op_run_info->requires_grad = GradState::Get().RequiresGrad();
  op_run_info->base_op_run_info.use_dynamic_shape_process = grad()->forward_use_dynamic_shape_process();
  PyNativeAlgo::PyParser::SetPrim(op_run_info, args[static_cast<size_t>(RunOpArgsEnum::PY_PRIM)]);
  OpRunInfoUsePrimC(op_run_info);
  PyNativeAlgo::PyParser::ParseOpInputByPythonObj(op_run_info, args[static_cast<size_t>(RunOpArgsEnum::PY_INPUTS)],
                                                  stub);
  op_run_info->base_op_run_info.device_target = GetCurrentDeviceTarget(op_run_info->op_grad_info->op_prim);
  bool is_dynamic_shape =
    op_run_info->base_op_run_info.has_dynamic_output || op_run_info->base_op_run_info.use_dynamic_shape_process;
  kernel::pyboost::PyBoostUtils::GetConstInputToAttr(
    op_run_info->op_grad_info->op_prim, op_run_info->base_op_run_info.op_name,
    op_run_info->base_op_run_info.device_target, is_dynamic_shape, &op_run_info->input_to_attr);
  bool is_dynamic_inputs = IsDynamicInputs(op_run_info);
  if (!op_run_info->input_to_attr.empty() || is_dynamic_inputs) {
    MS_LOG(DEBUG) << "Op_prim need clone:" << op_run_info->base_op_run_info.op_name
                  << ", is_dynamic_inputs:" << is_dynamic_inputs
                  << ", input_to_attr is not empty:" << (!op_run_info->input_to_attr.empty());
    ClonePrim(op_run_info);
  }
#ifndef ENABLE_TEST
  // Obtaining device context may fail in UT
  op_run_info->base_op_run_info.stream_id = CurrentStream::id();
#endif
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
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  VectorRef outputs;
  MS_EXCEPTION_IF_NULL(op_backend_);
  op_backend_->Run(backend_op_run_info, backend_op_run_info->base_op_run_info.device_target, device_id, &outputs);

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

ValuePtr ForwardExecutor::RunOpWithBackendPolicy(const FrontendOpRunInfoPtr &op_run_info,
                                                 const BackendOpRunInfoPtr &backend_op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
#ifndef ENABLE_TEST
  if (PyNativeAlgo::Common::IsVmOp(op_run_info->base_op_run_info.op_name)) {
    return RunOpInVM(op_run_info);
  }
  return RunOpInMs(op_run_info, backend_op_run_info);
#else
  return RunOpInVM(op_run_info);
#endif
}

ValuePtr ForwardExecutor::RunOpInVM(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_LOG(DEBUG) << "RunOpInVM start";
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->op_grad_info->run_in_vm = true;
  if (op_run_info->requires_grad) {
    for (size_t i = 0; i < op_run_info->input_size; i++) {
      op_run_info->op_grad_info->input_value_grad_type[i] =
        AutoGradUtil::SetValueGradInfo(op_run_info->op_grad_info->input_value[i], InputType::kConstant);
      (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(op_run_info->op_grad_info->input_value[i]);
    }
  }
  if (PyNativeAlgo::Common::IsVmOp(op_run_info->base_op_run_info.op_name)) {
    std::vector<ValuePtr> result(op_run_info->input_size);
    for (size_t i = 0; i < op_run_info->input_size; i++) {
      result[i] = CopyTensorValueWithNewId(op_run_info, op_run_info->op_grad_info->input_value[i]);
    }
    auto result_v = ConstructOutputInVM(result);
    if (op_run_info->requires_grad) {
      // Later we need modify hook op to view op.
      (void)AutoGradUtil::SetValueGradInfo(result_v, InputType::kOpOutput);
    }
    MS_LOG(DEBUG) << "RunOpInVM end";
    return result_v;
  }
  py::gil_scoped_acquire gil_acquire;
  MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info->op_prim);
  py::list vm_op_inputs = py::list(op_run_info->input_size);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    vm_op_inputs[i] = py::reinterpret_steal<py::object>(tensor::Wrap(op_run_info->op_grad_info->input_value[i]));
  }
  if (!utils::isa<PrimitivePy>(op_run_info->op_grad_info->op_prim)) {
    MS_LOG(EXCEPTION) << "Not a PrimitivePy, " << op_run_info->op_grad_info->op_prim->ToString();
  }
  auto result = utils::cast<PrimitivePyPtr>(op_run_info->op_grad_info->op_prim)->RunPyComputeFunction(vm_op_inputs);
  if (py::isinstance<py::none>(result)) {
    MS_LOG(EXCEPTION) << "VM op " << op_run_info->base_op_run_info.op_name << " run failed!";
  }
  ValuePtr result_v = parse::data_converter::PyObjToValue(result);
  if (!result_v->isa<ValueSequence>() && (op_run_info->base_op_run_info.abstract == nullptr ||
                                          op_run_info->base_op_run_info.abstract->isa<abstract::AbstractSequence>())) {
    result_v = std::make_shared<ValueTuple>(std::vector{result_v});
  }
  if (op_run_info->requires_grad) {
    (void)AutoGradUtil::SetValueGradInfo(result_v, InputType::kOpOutput);
  }
  MS_LOG(DEBUG) << "RunOpInVM end";
  return result_v;
}

bool ForwardExecutor::CellNotSetMixedPrecision(const PyboostOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (!mix_precision_type_stack_.empty() && mix_precision_type_stack_.top() != kNotSet) {
    op_run_info->mix_type = mix_precision_type_stack_.top();
    return false;
  }
  // get mix_precision_type from amp strategy stack
  return !GetMixprecisionTypeFromStrategy(op_run_info);
}

bool ForwardExecutor::CellNotSetMixedPrecision(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (!mix_precision_type_stack_.empty() && mix_precision_type_stack_.top() != kNotSet) {
    op_run_info->mix_type = mix_precision_type_stack_.top();
    return false;
  }
  // get mix_precision_type from amp strategy stack
  return !GetMixprecisionTypeFromStrategy(op_run_info);
}

void ForwardExecutor::ExecuteLazyTask() const {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kWaitPipeline);
  runtime::Pipeline::Get().WaitAll();
  MsException::Instance().CheckException();
}

device::DeviceType ForwardExecutor::GetCurrentDeviceTarget(const PrimitivePtr &op_prim) const {
  MS_EXCEPTION_IF_NULL(op_prim);
  PrimitiveReadLock read_lock(op_prim->shared_mutex());
  const auto &attr_map = op_prim->attrs();
  auto iter = attr_map.find("primitive_target");
  if (MS_UNLIKELY(iter != attr_map.end())) {
    return device::GetDeviceTypeByName(GetValue<std::string>(iter->second));
  }
  return DeviceManagerConf::GetInstance()->device_type();
}

void ForwardExecutor::Sync(const bool &) {
  GilReleaseWithCheck release_gil;
  ExecuteLazyTask();
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kSyncStream);
  device::DeviceContextManager::GetInstance().SyncAllStreams();
  runtime::ProfilerAnalyzer::GetInstance().EndStep();
  runtime::ProfilerAnalyzer::GetInstance().StartStep();
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
                                                  const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &device_address = input_tensor->device_address();
  const auto &device_context = runtime::OpRunner::GetDeviceContext(op_run_info->base_op_run_info.device_target);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_address->GetDeviceType() == device_context->GetDeviceType()) {
    // todo: copy data if the cpu device address is from python data.
    MS_LOG(DEBUG) << "Input device address is from memory pool, and device type is " << device_address->GetDeviceType()
                  << " device_target " << device_context->GetDeviceType();
    return;
  }

  MS_LOG(DEBUG) << "Input_tensor address is nullptr, need create address.";
  auto address_size = GetTypeByte(input_tensor->Dtype()) * static_cast<size_t>(input_tensor->ElementsNum());
  auto kernel_tensor = AnfAlgo::CreateKernelTensor(
    nullptr, address_size, Format::DEFAULT_FORMAT, input_tensor->data_type(), input_tensor->shape(),
    device::GetDeviceNameByType(device_context->device_context_key().device_type_),
    device_context->device_context_key().device_id_);
  kernel_tensor->SetType(std::make_shared<TensorType>(input_tensor->Dtype()));
  kernel_tensor->SetShape(std::make_shared<abstract::TensorShape>(input_tensor->shape()));
  kernel_tensor->set_stream_id(op_run_info->base_op_run_info.stream_id);

  input_tensor->set_implicit_copy_address(device_address);
  input_tensor->set_implicit_copy_storage_info(input_tensor->storage_info());
  input_tensor->set_implicit_copy_format(input_tensor->format());
  input_tensor->set_device_address(kernel_tensor->device_address());
  input_tensor->set_format(kernel_tensor->format());
  input_tensor->set_storage_info(kernel_tensor->tensor_storage_info());
  input_tensor->set_need_pipeline_sync(true);
  op_backend_->RunAllocMemTask(device_context, input_tensor, EnablePipeline(""));
}

DeviceAddressPtr ForwardExecutor::TensorContiguousCallback(const tensor::TensorPtr &input_tensor) {
  // Gil might be release  by ACL, so release here to reduce conflict
  const auto &storage_info = input_tensor->storage_info();
  const auto &device_addr = input_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_addr);
  if (storage_info == nullptr) {
    return device_addr;
  }
  // as_numpy sync promise contiguous run_sync
  return runtime::DeviceAddressUtils::ConvertContiguousDeviceAddress(nullptr, input_tensor);
}

void ForwardExecutor::PrepareOpInputs(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  PyNativeAlgo::DataConvert::GetInputTensor(op_run_info);
  for (const auto &value : op_run_info->base_op_run_info.expanded_input_values) {
    if (!value->isa<tensor::Tensor>()) {
      continue;
    }
  }
}

void ForwardExecutor::CreateViewOutputTensor(const FrontendOpRunInfoPtr &op_run_info,
                                             const tensor::TensorPtr &input_tensor,
                                             const TensorStorageInfoPtr &storage_info,
                                             runtime::KernelTaskType task_type, bool is_multi_output) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(storage_info);
  auto output_tensor = tensor::from_spec(input_tensor->data_type(), storage_info->shape, device::DeviceType::kNone);
  output_tensor->set_need_pipeline_sync(true);
  output_tensor->set_contiguous_callback(
    [this](const tensor::TensorPtr &self) -> DeviceAddressPtr { return TensorContiguousCallback(self); });

  auto input_device_address = input_tensor->device_address();
  MS_EXCEPTION_IF_NULL(input_device_address);
  if (task_type == runtime::KernelTaskType::kCOPY_TASK) {
    input_tensor->set_storage_info(storage_info);
  }

  // Create view output address

  auto kernel_tensor = AnfAlgo::CreateKernelTensor(nullptr, input_device_address->GetSize(), Format::DEFAULT_FORMAT,
                                                   output_tensor->data_type(), output_tensor->shape(),
                                                   device::GetDeviceNameByType(input_device_address->GetDeviceType()),
                                                   input_device_address->device_id());
  if (input_device_address->GetDeviceType() != device::DeviceType::kAscend) {
    // Not transmitting host shape information under Ascend for better performance.
    kernel_tensor->SetType(std::make_shared<TensorType>(TypeIdToType(output_tensor->data_type())));
    kernel_tensor->SetShape(std::make_shared<abstract::TensorShape>(output_tensor->shape()));
  }
  kernel_tensor->set_tensor_storage_info(storage_info);
  kernel_tensor->set_size(input_device_address->GetSize());
  kernel_tensor->set_stream_id(input_device_address->stream_id());

  auto output_device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(output_device_address);

  output_device_address->set_device_pointer(input_device_address->device_pointer());
  output_tensor->set_device_address(output_device_address);
  output_tensor->set_format(kernel_tensor->format());
  output_tensor->set_storage_info(kernel_tensor->tensor_storage_info());
  autograd::CreationType creationType =
    is_multi_output
      ? autograd::CreationType::kMultiOutput
      : (op_run_info->requires_grad ? autograd::CreationType::kDefault : autograd::CreationType::kNoGradMode);
  AutoGradUtil::BuildViewAutoGradMeta(input_tensor, output_tensor, creationType, op_run_info->requires_grad);
  (void)op_run_info->base_op_run_info.output_tensors.emplace_back(output_tensor);
}

ValuePtr ForwardExecutor::RunOpInMsInner(const FrontendOpRunInfoPtr &op_run_info,
                                         const BackendOpRunInfoPtr &backend_op_run_info) {
  const auto &outputs = RunOpBackendInner(op_run_info, backend_op_run_info);
  bool is_out_sequence = (op_run_info->base_op_run_info.abstract == nullptr ||
                          op_run_info->base_op_run_info.abstract->isa<abstract::AbstractSequence>());
  const auto &result_v = AutoGradUtil::VectorRefToValue(outputs, op_run_info->requires_grad, is_out_sequence);
  MS_LOG(DEBUG) << "RunOpInMs end";
  return result_v;
}

void ForwardExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear forward res";
  {
    GilReleaseWithCheck gil_release;
    runtime::Pipeline::Get().frontend_stage()->Clear();
  }
  runtime::OpExecutor::GetInstance().Reset();

  init_ = false;
  enable_async_ = false;
  last_target_ = "Unknown";
  std::stack<MixedPrecisionType>().swap(mix_precision_type_stack_);
  cast_operation()->ClearRes();
  ClearNodeAbsMap();
  infer_operation()->ClearPrimAbsList();
  infer_operation()->ClearConstFlagPrimCache();
  op_backend_ = std::make_unique<compile::OpBackend>();
  slice_prim_cache_.clear();
}

void ForwardExecutor::ChildAfterFork() {
  MS_LOG(DEBUG) << "ForwardExecutor reinitialize after fork.";
  op_backend_ = std::make_unique<compile::OpBackend>();
  MS_LOG(DEBUG) << "ForwardExecutor reinitialize after fork done.";
}
}  // namespace pynative
}  // namespace mindspore
