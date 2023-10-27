//
// Created by jojo on 2023/10/18.
//

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PY_BOOST_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PY_BOOST_UTILS_H_

#include "include/common/utils/tensor_future.h"
#include "runtime/pynative/op_executor.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
using DeviceAddressPromisePtr = pynative::DeviceAddressPromisePtr;
using DeviceAddressPromise = pynative::DeviceAddressPromise;
using DeviceAddressFutureDataPtr = pynative::DeviceAddressFutureDataPtr;
using DeviceAddressFuture = pynative::DeviceAddressFuture;
}  // namespace

class BACKEND_EXPORT PyBoostUtils {
 public:
  static void CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs);
};
KernelTensorPtr BACKEND_EXPORT TensorToKernelTensor(const TensorPtr &value, const DeviceContext *device_context);
KernelTensorPtr BACKEND_EXPORT ScalarToKernelTensor(const ScalarPtr &value, const DeviceContext *device_context);
std::vector<KernelTensorPtr> ValueToKernelTensor(const ValuePtrList &values, const DeviceContext *device_context);
DeviceContext *CreateOrGetDeviceContextAndInit(const std::string &target_device);
kernel::AddressPtrList CreateWorkspaceAddressForPyboostOp(std::vector<size_t> workspace_sizes,
                                                          const DeviceContext *device_context);
tensor::TensorPtr BACKEND_EXPORT ScalarToTensor(const ScalarPtr &scalar, const TypePtr &type);
tensor::TensorPtr BACKEND_EXPORT ContiguousTensor(const tensor::TensorPtr &input_tensor);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PY_BOOST_UTILS_H_
