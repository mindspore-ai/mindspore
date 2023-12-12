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
#include "kernel/common_utils.h"
#include "kernel/kernel_mod_cache.h"
#include "runtime/device/device_address_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_def.h"
#include "runtime/pynative/op_executor.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
using DeviceAddressPromisePtr = pynative::DeviceAddressPromisePtr;
using DeviceAddressPromise = pynative::DeviceAddressPromise;
using DeviceAddressFutureDataPtr = pynative::DeviceAddressFutureDataPtr;
using DeviceAddressFuture = pynative::DeviceAddressFuture;

namespace {
template <typename T>
tensor::TensorPtr CastToTensor(const ScalarPtr &scalar, const TypePtr &type) {
  if (scalar == nullptr) {
    MS_EXCEPTION(ArgumentError) << "Nullptr Error!";
  }
  TypePtr data_type = scalar->type();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();
  T value;
  switch (type_id) {
    case kNumberTypeBool:
      value = static_cast<T>(GetValue<bool>(scalar));
      break;
    case kNumberTypeInt8:
      value = static_cast<T>(GetValue<int8_t>(scalar));
      break;
    case kNumberTypeInt16:
      value = static_cast<T>(GetValue<int16_t>(scalar));
      break;
    case kNumberTypeInt32:
      value = static_cast<T>(GetValue<int32_t>(scalar));
      break;
    case kNumberTypeInt64:
      value = static_cast<T>(GetValue<int64_t>(scalar));
      break;
    case kNumberTypeUInt8:
      value = static_cast<T>(GetValue<uint8_t>(scalar));
      break;
    case kNumberTypeUInt16:
      value = static_cast<T>(GetValue<uint16_t>(scalar));
      break;
    case kNumberTypeUInt32:
      value = static_cast<T>(GetValue<uint32_t>(scalar));
      break;
    case kNumberTypeUInt64:
      value = static_cast<T>(GetValue<uint64_t>(scalar));
      break;
    case kNumberTypeFloat32:
      value = static_cast<T>(GetValue<float>(scalar));
      break;
    case kNumberTypeFloat64:
      value = static_cast<T>(GetValue<double>(scalar));
      break;
    default:
      MS_LOG(EXCEPTION) << "When convert scalar to tensor, the scalar type: " << data_type << " is invalid.";
  }
  return std::make_shared<tensor::Tensor>(value, type);
}
}  // namespace

void PyBoostUtils::CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs) {
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
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_type, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);
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
    nullptr, address_size, kOpFormat_DEFAULT, old_storage_info->data_type, old_storage_info->shape,
    device_context->device_context_key().device_name_, device_context->device_context_key().device_id_);
  auto new_device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
  new_device_address->set_device_shape(old_storage_info->shape);
  new_device_address->set_original_ref_count(SIZE_MAX);
  new_device_address->ResetRefCount();

  if (!device_context->GetKernelExecutor(false)->ExecuteKernelTask(
        pynative::KernelTaskType::kCONTIGUOUS_TASK, {old_device_address}, {old_storage_info}, {new_device_address})) {
    MS_LOG(EXCEPTION) << "ExecuteKernelTask failed, task_type:" << pynative::KernelTaskType::kCONTIGUOUS_TASK;
  }
  return new_device_address;
}

void PyBoostUtils::CreateOutputTensor(const tensor::TensorPtr &input, const TensorStorageInfoPtr &storage_info,
                                      std::vector<tensor::TensorPtr> *outputs) {
  auto output_tensor = std::make_shared<tensor::Tensor>(storage_info->data_type, storage_info->shape);
  output_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
  output_tensor->set_storage_info(storage_info);
  if (input->address_future()) {
    output_tensor->set_address_future(input->address_future());
  } else if (input->device_address()) {
    output_tensor->set_device_address(input->device_address());
  } else {
    MS_EXCEPTION_IF_NULL(input->device_address());
  }
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
  }
  MS_LOG(EXCEPTION) << "Cannot found infer function for Op " << prim->name();
}

device::DeviceAddressPtrList PyBoostUtils::CreateOutputDeviceAddress(DeviceContext *device_context,
                                                                     const abstract::AbstractBasePtr &abs,
                                                                     const std::vector<TensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(abs);

  auto output_size = outputs.size();
  device::DeviceAddressPtrList output_device_address;
  if (output_size == 1) {
    (void)output_device_address.emplace_back(
      runtime::DeviceAddressUtils::CreateOutputAddress(device_context, abs, kIndex0, outputs[kIndex0]));
  } else {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (output_size != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Outputs size " << output_size << " but output abstract size " << abs_seq->size();
    }
    for (size_t i = 0; i < output_size; ++i) {
      (void)output_device_address.emplace_back(
        runtime::DeviceAddressUtils::CreateOutputAddress(device_context, abs_seq->elements()[i], i, outputs[i]));
    }
  }
  return output_device_address;
}

tensor::TensorPtr PyBoostUtils::ScalarToTensor(const ScalarPtr &scalar, const TypePtr &type) {
  if (scalar == nullptr) {
    MS_EXCEPTION(ArgumentError) << "Nullptr Error!";
  }
  TypePtr data_type = scalar->type();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = type->type_id();
  switch (type_id) {
    case kNumberTypeBool:
      return CastToTensor<bool>(scalar, type);
    case kNumberTypeInt8:
      return CastToTensor<int8_t>(scalar, type);
    case kNumberTypeInt16:
      return CastToTensor<int16_t>(scalar, type);
    case kNumberTypeInt32:
      return CastToTensor<int32_t>(scalar, type);
    case kNumberTypeInt64:
      return CastToTensor<int64_t>(scalar, type);
    case kNumberTypeUInt8:
      return CastToTensor<uint8_t>(scalar, type);
    case kNumberTypeUInt16:
      return CastToTensor<uint16_t>(scalar, type);
    case kNumberTypeUInt32:
      return CastToTensor<uint32_t>(scalar, type);
    case kNumberTypeUInt64:
      return CastToTensor<uint64_t>(scalar, type);
    case kNumberTypeFloat32:
      return CastToTensor<float>(scalar, type);
    case kNumberTypeFloat64:
      return CastToTensor<double>(scalar, type);
    default:
      MS_LOG(EXCEPTION) << "When convert scalar to tensor, the dst type: " << type << " is invalid.";
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
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto sync = context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
  auto mode = context->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (sync || mode == mindspore::kGraphMode) {
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
  (void)kernel_tensor_list->emplace_back(device_address->kernel_tensor().get());
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

void PyBoostUtils::GetKernelTensor(DeviceContext *device_context, const abstract::AbstractBasePtr &input_abs,
                                   size_t index, std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                                   device::DeviceAddressPtrList *device_address_list,
                                   const std::optional<tensor::TensorPtr> &tensor) {
  if (tensor.has_value()) {
    GetKernelTensor(device_context, input_abs, index, kernel_tensor_list, device_address_list, tensor.value());
  } else {
    MS_EXCEPTION_IF_NULL(kernel_tensor_list);
    MS_EXCEPTION_IF_NULL(device_address_list);
    (void)device_address_list->emplace_back(nullptr);
    (void)kernel_tensor_list->emplace_back(nullptr);
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
    auto kernel_tensor = std::make_shared<KernelTensor>(nullptr, workspace_size, "", kTypeUnknown, ShapeVector(),
                                                        device_context->device_context_key().device_name_,
                                                        device_context->device_context_key().device_id_);
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
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
