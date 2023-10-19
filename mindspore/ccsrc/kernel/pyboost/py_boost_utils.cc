//
// Created by jojo on 2023/10/18.
//

#include "py_boost_utils.h"

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
KernelTensorPtr TensorToKernelTensor(const ValuePtr &value,const DeviceContext *device_context) {
  if (!value->isa<tensor::Tensor>()) {
    MS_EXCEPTION(TypeError) << "value is not Tensor";
  }
  auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(value);
  auto &shape = tensor->shape();
  auto dtype = tensor->Dtype();
  size_t tensor_size = SizeOf(shape) * GetDataTypeSize(dtype->type_id());

  // TODO (CARRY) Waiting dyn_shape_dev
  //  auto new_kernel_tensor = std::make_shared<kernel::KernelTensor>(nullptr, tensor_size, tensor->device_info().host_format_, dtype, shape,
  //                                                                  device_context->device_context_key().device_name_,
  //                                                                  device_context->device_context_key().device_id_);
  //  return new_kernel_tensor;
  return std::make_shared<KernelTensor>();
}
KernelTensorPtr ScalarToKernelTensor(const ValuePtr &value,const DeviceContext *device_context) {
  if (!value->isa<Scalar>()) {
    MS_EXCEPTION(TypeError) << "value is not Scalar";
  }
  auto scalar = std::dynamic_pointer_cast<Scalar>(value);
  auto dtype = scalar->type();
  size_t tensor_size = GetDataTypeSize(dtype->type_id());

  // TODO (CARRY) Waiting dyn_shape_dev
  //  auto new_kernel_tensor = std::make_shared<kernel::KernelTensor>(nullptr, tensor_size, tensor->device_info().host_format_, dtype, {},
  //                                                                  device_context->device_context_key().device_name_,
  //                                                                  device_context->device_context_key().device_id_);
  //  return new_kernel_tensor;
  return std::make_shared<KernelTensor>();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
