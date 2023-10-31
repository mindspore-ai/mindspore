//
// Created by jojo on 2023/10/31.
//

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_BIAS_ADD_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_BIAS_ADD_H_

#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr BiasAddAscendCall(const PrimitivePtr &primitive, const tensor::TensorPtr &input_x,
                                    const tensor::TensorPtr &bias, const tensor::TensorPtr &output);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_BIAS_ADD_H_
