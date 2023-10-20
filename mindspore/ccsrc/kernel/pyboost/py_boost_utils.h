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
template <typename T, std::size_t... ls>
AbstractBasePtr InferImpl(const PrimitivePtr &primitive, const T &t, std::index_sequence<ls...>) {
  auto eval_impl = abstract::GetPrimitiveInferImpl(primitive);
  std::vector<AbstractBasePtr> input_abs;
  [&input_abs, &t]() { (input_abs.emplace_back(std::get<ls>(t)->ToAbstract()), ...); }();
  return eval_impl->InferShapeAndType(nullptr, primitive, input_abs);
}

template <typename... T>
void BACKEND_EXPORT InferOutput(const PrimitivePtr &primitive, std::vector<tensor::TensorPtr> *outputs,
                                const std::tuple<T...> &t) {
  auto output_abs = InferImpl(primitive, t, std::index_sequence_for<T...>());
  outputs->clear();
  PyBoostUtils::CreateOutputTensor(output_abs, outputs);
}
KernelTensorPtr TensorToKernelTensor(const TensorPtr &value, const DeviceContext *device_context);
KernelTensorPtr ScalarToKernelTensor(const ScalarPtr &value, const DeviceContext *device_context);
tensor::TensorPtr ScalarToTensor(const ScalarPtr &scalar, const TypePtr &type);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PY_BOOST_UTILS_H_
