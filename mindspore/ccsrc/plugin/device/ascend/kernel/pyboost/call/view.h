//
// Created by jojo on 2023/11/1.
//

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_VIEW_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_VIEW_H_
#include "ir/tensor.h"
#include "ir/value.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ViewAscendCall(const tensor::TensorPtr &input_tensor, const ValueTuplePtr &input_perm);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_VIEW_H_
