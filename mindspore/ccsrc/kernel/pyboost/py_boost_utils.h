//
// Created by jojo on 2023/10/18.
//

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PY_BOOST_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PY_BOOST_UTILS_H_

#include "include/common/utils/tensor_future.h"
#include "runtime/pynative/op_executor.h"
#include "mindspore/core/ops/view/view_strides_calculator.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class BACKEND_EXPORT PyBoostUtils {
 public:
  static void CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs);
  static DeviceContext *GetDeviceContext(const std::string &device_type);
  static void CreateOutputTensor(const tensor::TensorPtr &input, const TensorStorageInfoPtr &storage_info,
                                 std::vector<tensor::TensorPtr> *outputs);
  static AbstractBasePtr InferByOpDef(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_abs);
};
KernelTensorPtr BACKEND_EXPORT TensorToKernelTensor(const TensorPtr &value, const DeviceContext *device_context);
KernelTensorPtr BACKEND_EXPORT ScalarToKernelTensor(const ScalarPtr &value, const DeviceContext *device_context);
std::vector<KernelTensorPtr> ValueToKernelTensor(const ValuePtrList &values, const DeviceContext *device_context);
DeviceContext *CreateOrGetDeviceContextAndInit(const std::string &target_device);
kernel::AddressPtrList CreateWorkspaceAddressForPyboostOp(std::vector<size_t> workspace_sizes,
                                                          const DeviceContext *device_context);
tensor::TensorPtr BACKEND_EXPORT ScalarToTensor(const ScalarPtr &scalar, const TypePtr &type);
tensor::TensorPtr BACKEND_EXPORT ContiguousTensor(const tensor::TensorPtr &input_tensor);

template <typename... Args>
void PrepareOpInputs(DeviceContext *device_context, Args... args) {
  [&]() { (runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, args, "input"), ...); }();
}

void BACKEND_EXPORT PrepareOpOutputs(DeviceContext *device_context, const std::vector<TensorPtr> &outputs);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PY_BOOST_UTILS_H_
