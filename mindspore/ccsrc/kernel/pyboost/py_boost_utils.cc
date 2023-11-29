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
#include "kernel/common_utils.h"
#include "runtime/device/device_address_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_def.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
using DeviceAddressPromisePtr = pynative::DeviceAddressPromisePtr;
using DeviceAddressPromise = pynative::DeviceAddressPromise;
using DeviceAddressFutureDataPtr = pynative::DeviceAddressFutureDataPtr;
using DeviceAddressFuture = pynative::DeviceAddressFuture;

void PyBoostUtils::CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs,
                                      std::vector<pynative::DeviceAddressPromisePtr> *device_sync_promises) {
  auto create_tensor = [&outputs, &device_sync_promises](const TypePtr &type, const ShapeVector &shape_vector) {
    auto output_tensor = std::make_shared<tensor::Tensor>(type->type_id(), shape_vector);
    output_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
    (void)outputs->emplace_back(output_tensor);
    MS_LOG(DEBUG) << "Create output tensor " << output_tensor->ToString();

    DeviceAddressPromisePtr promise =
      std::make_shared<DeviceAddressPromise>(std::promise<DeviceAddressFutureDataPtr>());
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

device::DeviceAddressPtr ContiguousByDeviceAddress(const device::DeviceAddressPtr &old_device_address,
                                                   const TensorStorageInfoPtr &old_storage_info) {
  MS_EXCEPTION_IF_NULL(old_device_address);
  MS_EXCEPTION_IF_NULL(old_storage_info);

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
  MS_LOG(DEBUG) << prim->name() << " infer failed";
  return nullptr;
}

std::vector<device::DeviceAddressPtr> PrepareOpOutputs(DeviceContext *device_context,
                                                       const std::vector<TensorPtr> &outputs) {
  std::vector<device::DeviceAddressPtr> output_device_address;
  for (const auto &output : outputs) {
    (void)output_device_address.emplace_back(
      runtime::DeviceAddressUtils::CreateInputAddress(device_context, output, "output"));
  }
  return output_device_address;
}

std::vector<device::DeviceAddressPtr> PrepareOpOutputs(
  DeviceContext *device_context, const std::vector<TensorPtr> &outputs,
  const std::vector<pynative::DeviceAddressPromisePtr> &device_sync_promises) {
  auto output_size = outputs.size();
  if (output_size != device_sync_promises.size()) {
    MS_LOG(EXCEPTION) << "outputs size " << output_size << " but device_sync_promises size "
                      << device_sync_promises.size();
  }
  std::vector<device::DeviceAddressPtr> output_device_address;
  for (size_t i = 0; i < output_size; ++i) {
    (void)output_device_address.emplace_back(runtime::DeviceAddressUtils::CreateOutputTensorAddress(
      device_context, outputs[i], device_sync_promises[i], "output"));
  }
  return output_device_address;
}

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

tensor::TensorPtr ScalarToTensor(const ScalarPtr &scalar, const TypePtr &type) {
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

tensor::TensorPtr ContiguousTensor(const tensor::TensorPtr &input_tensor) {
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

DeviceContext *CreateOrGetDeviceContextAndInit(const std::string &target_device) {
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {target_device, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);
  return device_context;
}

void DispatchRun(const std::shared_ptr<pynative::PyBoostDeviceTask> &task) { task->Run(); }

std::vector<kernel::KernelTensor *> GetWorkspaceKernelTensors(const KernelModPtr &kernel_mod,
                                                              const device::DeviceContext *device_context,
                                                              const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  const auto &workspace_sizes = kernel_mod->GetWorkspaceSizeList();
  std::vector<device::DeviceAddressPtr> add_workspaces;
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto kernel_tensor = std::make_shared<KernelTensor>(nullptr, workspace_sizes[i], "", kTypeUnknown, ShapeVector(),
                                                        device_context->device_context_key().device_name_,
                                                        device_context->device_context_key().device_id_);
    auto device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
    MS_LOG(DEBUG) << "Create workspace for op: " << op_name << " addr: " << device_address;
    MS_EXCEPTION_IF_NULL(device_address);
    (void)add_workspaces.emplace_back(device_address);
  }

  std::vector<kernel::KernelTensor *> workspaces;
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto device_address = add_workspaces[i];
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed";
    }
    (void)workspaces.emplace_back(device_address->kernel_tensor().get());
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << workspaces.back()->device_ptr()
                  << " size:" << workspaces.back()->size();
  }
  return workspaces;
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
