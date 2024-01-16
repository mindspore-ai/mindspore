/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/pyboost/py_boost_utils.h"
#include <algorithm>
#include <utility>
#include <unordered_map>
#include "kernel/common_utils.h"
#include "kernel/kernel_mod_cache.h"
#include "runtime/device/device_address_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_def.h"
#include "runtime/pynative/op_executor.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "mindspore/ccsrc/plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/pyboost/ops/cast.h"
#include "mindspore/core/ops/array_ops.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
using DeviceAddressPromisePtr = pynative::DeviceAddressPromisePtr;
using DeviceAddressPromise = pynative::DeviceAddressPromise;
using DeviceAddressFutureDataPtr = pynative::DeviceAddressFutureDataPtr;
using DeviceAddressFuture = pynative::DeviceAddressFuture;

void PyBoostUtils::CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                     runtime::ProfilerEvent::kPyBoostCreateOutputTensor,
                                     runtime::ProfilerRecorder::kNoName, false);
  auto create_tensor = [&outputs](const TypePtr &type, const ShapeVector &shape_vector) {
    auto output_tensor = std::make_shared<tensor::Tensor>(type->type_id(), shape_vector);
    output_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
    (void)outputs->emplace_back(output_tensor);
    MS_LOG(DEBUG) << "Create output tensor " << output_tensor->ToString();
  };

  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<abstract::AbstractSequence>()) {
    auto seq = abstract->cast<abstract::AbstractSequencePtr>();
    auto elements = seq->elements();
    for (const auto &element : elements) {
      CreateOutputTensor(element, outputs);
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

DeviceContext *PyBoostUtils::GetDeviceContext(const std::string &device_type) {
  static std::unordered_map<std::string, DeviceContext *> device_contexts;
  auto iter = device_contexts.find(device_type);
  if (iter != device_contexts.end()) {
    return iter->second;
  }

  auto device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_type, device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);
  device_contexts[device_type] = device_context;
  MS_LOG(DEBUG) << "Get device context of " << device_type << " id " << device_id;
  return device_context;
}

kernel::KernelModPtr PyBoostUtils::CreateKernelMod(const PrimitivePtr &prim, const std::string &op_name,
                                                   DeviceContext *device_context,
                                                   const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &device_name = device_context->device_context_key().device_name_;

  auto &cache_helper = kernel::KernelModCache::GetInstance();
  const auto &key = cache_helper.GetKernelModKey(op_name, device_name, inputs);
  auto kernel_mod = cache_helper.GetKernelMod(key);
  if (kernel_mod == nullptr) {
    kernel_mod = device_context->GetKernelExecutor(false)->CreateKernelMod(op_name);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    if (!kernel_mod->Init(prim, inputs, outputs)) {
      MS_LOG(EXCEPTION) << "KernelMod Init Failed: " << op_name;
    }
    cache_helper.SetCache(key, kernel_mod);
    PyboostKernelExtraFuncFactory::GetInstance().SetThreadPool(device_name, kernel_mod);
  }

  return kernel_mod;
}

device::DeviceAddressPtr PyBoostUtils::ContiguousByDeviceAddress(const device::DeviceAddressPtr &old_device_address,
                                                                 const TensorStorageInfoPtr &old_storage_info) {
  MS_EXCEPTION_IF_NULL(old_device_address);
  MS_EXCEPTION_IF_NULL(old_storage_info);
  GilReleaseWithCheck gil_release;

  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {old_device_address->device_name(), old_device_address->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);

  auto address_size = GetTypeByte(TypeIdToType(old_device_address->type_id())) * SizeOf(old_storage_info->shape);
  if (old_storage_info->data_type == kTypeUnknown) {
    MS_LOG(EXCEPTION) << "The view op out type is kTypeUnknown";
  }

  auto kernel_tensor = std::make_shared<kernel::KernelTensor>(
    nullptr, address_size, Format::DEFAULT_FORMAT, old_storage_info->data_type, old_storage_info->shape,
    device_context->device_context_key().device_name_, device_context->device_context_key().device_id_);
  auto new_device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
  new_device_address->set_device_shape(old_storage_info->shape);
  new_device_address->set_original_ref_count(SIZE_MAX);
  new_device_address->ResetRefCount();
  auto stream_id = device_context->device_res_manager_->GetCurrentStreamId();

  if (!device_context->GetKernelExecutor(false)->ExecuteKernelTask(pynative::KernelTaskType::kCONTIGUOUS_TASK,
                                                                   {old_device_address}, {old_storage_info},
                                                                   {new_device_address}, stream_id)) {
    MS_LOG(EXCEPTION) << "ExecuteKernelTask failed, task_type:" << pynative::KernelTaskType::kCONTIGUOUS_TASK;
  }
  return new_device_address;
}

void PyBoostUtils::CreateOutputTensor(const tensor::TensorPtr &input, const TensorStorageInfoPtr &storage_info,
                                      std::vector<tensor::TensorPtr> *outputs) {
  auto output_tensor = std::make_shared<tensor::Tensor>(storage_info->data_type, storage_info->shape);
  output_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
  output_tensor->set_storage_info(storage_info);
  output_tensor->set_device_address(input->device_address());
  output_tensor->set_contiguous_callback([](const tensor::TensorPtr &tensor, const DeviceSyncPtr &device_address,
                                            const TensorStorageInfoPtr &storage_info) -> DeviceSyncPtr {
    if (tensor != nullptr) {
      ContiguousTensor(tensor);
      return nullptr;
    }

    auto device_addr = std::dynamic_pointer_cast<device::DeviceAddress>(device_address);
    return ContiguousByDeviceAddress(device_addr, storage_info);
  });
  (void)outputs->emplace_back(output_tensor);
  MS_LOG(DEBUG) << "Create output tensor " << output_tensor->ToString();
}

AbstractBasePtr PyBoostUtils::InferByOpDef(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_abs) {
  MS_EXCEPTION_IF_NULL(prim);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostInferByOpDef,
                                     prim->name(), false);
  auto frontend_func_impl = mindspore::ops::GetOpFrontendFuncImplPtr(prim->name());
  AbstractBasePtr output_abs = nullptr;
  if (frontend_func_impl) {
    output_abs = frontend_func_impl->InferAbstract(prim, input_abs);
    if (output_abs != nullptr) {
      MS_LOG(DEBUG) << "Pynative Infer by InferAbstract, got abstract: " << output_abs->ToString();
      return output_abs;
    }
  }

  auto op_def = mindspore::ops::GetOpDef(prim->name());
  if (op_def) {
    (void)op_def->func_impl_.CheckValidation(prim, input_abs);
    auto shape = op_def->func_impl_.InferShape(prim, input_abs);
    auto type = op_def->func_impl_.InferType(prim, input_abs);
    output_abs = mindspore::abstract::MakeAbstract(shape, type);
    MS_LOG(DEBUG) << "Pynative Infer by OpDef, got abstract: " << output_abs->ToString();
    return output_abs;
  } else {
    const auto &infer_map = abstract::GetPrimitiveInferMapPtr();
    const auto &iter = infer_map->find(prim);
    if (iter != infer_map->end()) {
      output_abs = iter->second.InferShapeAndType(nullptr, prim, input_abs);
      MS_LOG(DEBUG) << "Pynative Infer by C++ PrimitiveInferMap, got abstract: " << output_abs->ToString();
      return output_abs;
    } else {
      MS_LOG(EXCEPTION) << "Cannot found infer function for Op " << prim->name();
    }
  }
}

tensor::TensorPtr PyBoostUtils::ContiguousTensor(const tensor::TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  if (input_tensor->storage_info() == nullptr) {
    return input_tensor;
  }
  auto old_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
  MS_EXCEPTION_IF_NULL(old_device_address);
  auto old_storage_info = input_tensor->storage_info();
  auto new_device_address = ContiguousByDeviceAddress(old_device_address, old_storage_info);

  input_tensor->set_device_address(new_device_address);
  input_tensor->set_storage_info(nullptr);
  return input_tensor;
}

DeviceContext *PyBoostUtils::CreateOrGetDeviceContextAndInit(const std::string &target_device) {
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {target_device, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);
  return device_context;
}

void PyBoostUtils::DispatchRun(const std::shared_ptr<pynative::PyBoostDeviceTask> &task) {
  static auto need_sync = runtime::OpExecutor::NeedSync();
  if (need_sync) {
    MS_LOG(INFO) << "PyBoost sync run device task";
    runtime::OpExecutor::GetInstance().WaitAll();
    task->Run();
  } else {
    runtime::OpExecutor::GetInstance().PushOpRunTask(task);
  }
}

std::vector<kernel::KernelTensor *> PyBoostUtils::GetKernelTensorFromAddress(
  const device::DeviceAddressPtrList &input_device_address) {
  std::vector<kernel::KernelTensor *> input_kernel_tensors;
  std::transform(input_device_address.begin(), input_device_address.end(), std::back_inserter(input_kernel_tensors),
                 [](const auto &item) { return item->kernel_tensor().get(); });
  return input_kernel_tensors;
}

void PyBoostUtils::GetKernelTensor(DeviceContext *device_context, const abstract::AbstractBasePtr &input_abs,
                                   size_t index, std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                                   device::DeviceAddressPtrList *device_address_list, const TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor_list);
  MS_EXCEPTION_IF_NULL(device_address_list);

  const auto &device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  (void)device_address_list->emplace_back(device_address);
  const auto &kernel_tensor = device_address->kernel_tensor();
  (void)kernel_tensor_list->emplace_back(kernel_tensor.get());
  if (!kernel_tensor->host_info_exist()) {
    kernel_tensor->SetHostInfo(std::make_shared<abstract::TensorShape>(tensor->shape()),
                               std::make_shared<TensorType>(tensor->Dtype()), nullptr);
  }
}

void PyBoostUtils::GetKernelTensor(DeviceContext *device_context, const abstract::AbstractBasePtr &input_abs,
                                   size_t index, std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                                   device::DeviceAddressPtrList *device_address_list,
                                   const std::vector<TensorPtr> &tensors) {
  for (const auto &tensor : tensors) {
    // input_abs is not used in GetKernelTensor when value is TensorPtr.
    GetKernelTensor(device_context, input_abs, index, kernel_tensor_list, device_address_list, tensor);
  }
}

device::DeviceAddressPtrList PyBoostUtils::CreateWorkSpaceDeviceAddress(const KernelModPtr &kernel_mod,
                                                                        const device::DeviceContext *device_context,
                                                                        const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  const auto &workspace_sizes = kernel_mod->GetWorkspaceSizeList();
  device::DeviceAddressPtrList workspaces_address;
  for (const auto workspace_size : workspace_sizes) {
    auto kernel_tensor = std::make_shared<KernelTensor>(
      nullptr, workspace_size, Format::DEFAULT_FORMAT, kTypeUnknown, ShapeVector(),
      device_context->device_context_key().device_name_, device_context->device_context_key().device_id_);
    auto device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
    MS_LOG(DEBUG) << "Create workspace for op: " << op_name << " addr: " << device_address;
    MS_EXCEPTION_IF_NULL(device_address);
    (void)workspaces_address.emplace_back(device_address);
  }

  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto device_address = workspaces_address[i];
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed";
    }
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << device_address->kernel_tensor()->device_ptr()
                  << " size:" << device_address->kernel_tensor()->size();
  }
  return workspaces_address;
}

PyboostKernelExtraFuncFactory &PyboostKernelExtraFuncFactory::GetInstance() {
  static PyboostKernelExtraFuncFactory instance;
  return instance;
}

void PyBoostUtils::LaunchKernel(const PrimitivePtr &primitive, device::DeviceContext *device_context,
                                const AddressInfoPair &input_address_info, const AddressInfoPair &output_address_info,
                                void *stream_ptr) {
  const auto &real_name = primitive->name();
  // KernelMod init
  auto kernel_mod = PyBoostUtils::CreateKernelMod(primitive, real_name, device_context, input_address_info.first,
                                                  output_address_info.first);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  // KernelMod resize
  if (kernel_mod->Resize(input_address_info.first, output_address_info.first) == kernel::KRET_RESIZE_FAILED) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#CPU kernel op [" << real_name << "] resize failed.";
  }
  // Get workspace address
  const auto &workspace_device_address =
    PyBoostUtils::CreateWorkSpaceDeviceAddress(kernel_mod, device_context, primitive->name());
  const auto &workspace_kernel_tensors = PyBoostUtils::GetKernelTensorFromAddress(workspace_device_address);
  // Do kernel launch
  if (!kernel_mod->Launch(input_address_info.first, workspace_kernel_tensors, output_address_info.first, stream_ptr)) {
    MS_LOG(EXCEPTION) << "Launch kernel failed, name: " << real_name;
  }
  MS_LOG(DEBUG) << real_name << " Launch end";
}

TypeId GetTypeIdFromAbstractTensor(const AbstractBasePtr &abs_base) {
  if (abs_base->isa<abstract::AbstractTensor>()) {
    auto abs_tensor = std::dynamic_pointer_cast<abstract::AbstractTensor>(abs_base);
    return abs_tensor->element()->BuildType()->type_id();
  }
  return abs_base->BuildType()->type_id();
}

std::vector<TypeId> GetTypeFromAbstractBase(const AbstractBasePtr &abs_base) {
  if (abs_base->isa<abstract::AbstractTuple>()) {
    auto abs_tuple = std::dynamic_pointer_cast<abstract::AbstractTuple>(abs_base);
    std::vector<TypeId> input_type;
    for (auto &abs : abs_tuple->elements()) {
      (void)input_type.emplace_back(GetTypeIdFromAbstractTensor(abs));
    }
    return input_type;
  } else {
    const auto &type_id = GetTypeIdFromAbstractTensor(abs_base);
    return {type_id};
  }
}

std::vector<TypeId> GetTypeFromAbstractBase(const std::vector<AbstractBasePtr> &abs_vec) {
  std::vector<TypeId> input_type;
  for (auto &abs : abs_vec) {
    if (abs->isa<abstract::AbstractTuple>()) {
      // a tuple tensors have same type
      auto abs_tuple = std::dynamic_pointer_cast<abstract::AbstractTuple>(abs);
      input_type.emplace_back(abs_tuple->elements()[0]->BuildType()->type_id());
    } else {
      input_type.emplace_back(GetTypeIdFromAbstractTensor(abs));
    }
  }
  return input_type;
}

bool InputDtypeMatch(TypeId input_attr, TypeId input_type) {
  if (input_attr == input_type || kTypeUnknown == input_type) {
    return true;
  }
  if (input_attr == kNumberTypeInt32 && (input_type == kNumberTypeInt16 || input_type == kNumberTypeInt64)) {
    return true;
  }
  if (input_attr == kNumberTypeFloat32 && (input_type == kNumberTypeFloat16 || input_type == kNumberTypeFloat64)) {
    return true;
  }
  return false;
}

bool IsObjectTypeWeaklyMatched(const std::vector<TypeId> &object_dtypes,
                               const std::vector<DataType> &kernel_data_types) {
  // only support CPU
  for (size_t i = 0; i < object_dtypes.size(); i++) {
    // For optional input, the real input object type can be a None.
    if (!InputDtypeMatch(kernel_data_types[i].dtype, object_dtypes[i])) {
      return false;
    }
  }
  return true;
}

bool IsObjectTypeStrictlyMatched(const std::vector<TypeId> &object_dtypes,
                                 const std::vector<DataType> &kernel_data_types) {
  if (object_dtypes.size() != kernel_data_types.size()) {
    return false;
  }

  for (size_t i = 0; i < object_dtypes.size(); i++) {
    // For optional input, the real input object type can be a None.
    if (object_dtypes[i] != kernel_data_types[i].dtype) {
      return false;
    }
  }

  return true;
}

std::pair<bool, KernelAttr> PyBoostUtils::SelectKernel(const std::vector<AbstractBasePtr> &inputs_abs,
                                                       const AbstractBasePtr &outputs_abs,
                                                       DeviceContext *device_context, const std::string &op_name) {
  // only support CPU
  const auto &kernel_mod = device_context->GetKernelExecutor(false)->CreateKernelMod(op_name);
  const auto &support_list = kernel_mod->GetOpSupport();
  const auto &inputs_object_dtypes = GetTypeFromAbstractBase(inputs_abs);
  const auto &output_object_dtypes = GetTypeFromAbstractBase(outputs_abs);
  for (auto &cur_kernel_attr : support_list) {
    auto data_pair = kernel::GetInOutDataTypesFromKernelAttr(cur_kernel_attr);
    const auto &[input_data_types, output_data_types] = kernel::GetInOutDataTypesFromKernelAttr(cur_kernel_attr);
    if (IsObjectTypeStrictlyMatched(inputs_object_dtypes, input_data_types) &&
        IsObjectTypeStrictlyMatched(output_object_dtypes, output_data_types)) {
      return std::make_pair(true, cur_kernel_attr);
    }

    if (IsObjectTypeWeaklyMatched(inputs_object_dtypes, input_data_types) &&
        IsObjectTypeWeaklyMatched(output_object_dtypes, output_data_types)) {
      return std::make_pair(false, cur_kernel_attr);
    }
  }
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  for (auto &input_type : inputs_object_dtypes) {
    (void)inputs.emplace_back(TypeIdToString(input_type));
  }
  for (auto &output_type : output_object_dtypes) {
    (void)outputs.emplace_back(TypeIdToString(output_type));
  }
  MS_LOG(EXCEPTION) << "Unsupported op [" << op_name << "] on CPU, input_type:" << inputs << " ,output_type:" << outputs
                    << ". Please confirm whether the device target setting is correct, "
                    << "or refer to 'mindspore.ops' at https://www.mindspore.cn to query the operator support list.";
}

tensor::TensorPtr PyBoostUtils::CastTensor(const tensor::TensorPtr &tensor, const TypeId &type_id,
                                           const std::string &device_target) {
  if (tensor->Dtype()->type_id() == type_id) {
    return tensor;
  }
  const auto &cast_op = CREATE_PYBOOST_OP(Cast, device_target);
  cast_op->set_primitive(prim::kPrimCast);
  return cast_op->Call(tensor, TypeIdToType(type_id));
}

std::vector<tensor::TensorPtr> PyBoostUtils::CastTensor(const std::vector<tensor::TensorPtr> &tensors,
                                                        const std::vector<TypeId> &type_id_list,
                                                        const std::string &device_target) {
  if (tensors.size() != type_id_list.size()) {
    MS_LOG(EXCEPTION) << "before cast tensor output size is not equal after cast";
  }
  std::vector<tensor::TensorPtr> output_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &output = CastTensor(tensors[i], type_id_list[i], device_target);
    (void)output_tensors.emplace_back(output);
  }
  return output_tensors;
}

std::vector<tensor::TensorPtr> PyBoostUtils::CastTensor(const std::vector<tensor::TensorPtr> &tensors, TypeId type_id,
                                                        const std::string &device_target) {
  // tuple input
  std::vector<tensor::TensorPtr> output_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &output = CastTensor(tensors[i], type_id, device_target);
    (void)output_tensors.emplace_back(output);
  }
  return output_tensors;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
