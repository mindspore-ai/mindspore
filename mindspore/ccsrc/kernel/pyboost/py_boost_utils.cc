//
// Created by jojo on 2023/10/18.
//

#include "py_boost_utils.h"
#include "kernel/common_utils.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
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
KernelTensorPtr TensorToKernelTensor(const TensorPtr &tensor, const DeviceContext *device_context) {
  // TODO (CARRY) Waiting dyn_shape_dev
  //  auto new_kernel_tensor = std::make_shared<kernel::KernelTensor>(nullptr, tensor_size,
  //  tensor->device_info().host_format_, dtype, shape,
  //                                                                  device_context->device_context_key().device_name_,
  //                                                                  device_context->device_context_key().device_id_);
  //  return new_kernel_tensor;
  auto kernel_tensor = std::make_shared<KernelTensor>(tensor);
  return kernel_tensor;
}
KernelTensorPtr ScalarToKernelTensor(const ScalarPtr &scalar, const DeviceContext *device_context) {
  // TODO (CARRY) Waiting dyn_shape_dev
  //  auto new_kernel_tensor = std::make_shared<kernel::KernelTensor>(nullptr, tensor_size,
  //  tensor->device_info().host_format_, dtype, {},
  //                                                                  device_context->device_context_key().device_name_,
  //                                                                  device_context->device_context_key().device_id_);
  //  return new_kernel_tensor;
  return std::make_shared<KernelTensor>(scalar);
}

std::vector<KernelTensorPtr> ValueToKernelTensor(const ValuePtrList &values, const DeviceContext *device_context) {
  std::vector<KernelTensorPtr> kernel_tensors;
  for (auto &value : values) {
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<Scalar>()) {
      auto scalar = std::dynamic_pointer_cast<Scalar>(value);
      kernel_tensors.emplace_back(std::make_shared<KernelTensor>(scalar));
    } else if (value->isa<tensor::Tensor>()) {
      auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(value);
      kernel_tensors.emplace_back(std::make_shared<KernelTensor>(tensor));
    } else {
      MS_EXCEPTION(TypeError) << "value type is not supported";
    }
  }
  return kernel_tensors;
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
kernel::AddressPtrList CreateWorkspaceAddressForPyboostOp(std::vector<size_t> workspace_sizes,
                                                          const DeviceContext *device_context) {
  kernel::AddressPtrList workspaces;
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto workspace_device_address =
      runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context, workspace_sizes[i]);
    (void)workspaces.emplace_back(std::make_shared<kernel::Address>(workspace_device_address->GetMutablePtr(),
                                                                    workspace_device_address->GetSize()));
  }
  return workspaces;
}

DeviceContext *CreateOrGetDeviceContextAndInit(const std::string &target_device) {
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kGPUDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);
  return device_context;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
