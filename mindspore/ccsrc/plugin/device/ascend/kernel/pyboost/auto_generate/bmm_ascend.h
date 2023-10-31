//
// Created by jojo on 2023/10/30.
//

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_BMM_ASCEND_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_BMM_ASCEND_H_

#include "kernel/pyboost/auto_generate/bmm.h"
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class BmmAscend : public pyboost::Bmm {
 public:
  BmmAscend() = default;
  ~BmmAscend() = default;

  tensor::TensorPtr Call(const tensor::TensorPtr &input, const tensor::TensorPtr &mat2) override;
};

MS_REG_PYBOOST_OP(Ascend, Bmm);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_BMM_ASCEND_H_
