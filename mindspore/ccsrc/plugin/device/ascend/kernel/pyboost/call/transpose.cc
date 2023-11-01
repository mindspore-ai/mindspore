//
// Created by jojo on 2023/11/1.
//
#include "plugin/device/ascend/kernel/pyboost/call/transpose.h"
namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr TransposeAscendCall(const tensor::TensorPtr &input_tensor, const ValueTuplePtr &input_perm) {
  MS_LOG(EXCEPTION) << "Not impl";
  return nullptr;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore