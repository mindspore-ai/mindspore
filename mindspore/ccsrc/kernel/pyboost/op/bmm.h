//
// Created by jojo on 2023/10/30.
//

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_BMM_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_BMM_H_

#include "kernel/pyboost/op_register.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class BACKEND_EXPORT Bmm : public pyboost::Op {
 public:
  Bmm() = default;
  ~Bmm() = default;

  void CastInput() override {}
  virtual tensor::TensorPtr Call(const tensor::TensorPtr &input, const tensor::TensorPtr &mat2) = 0;
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_BMM_H_
