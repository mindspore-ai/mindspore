/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_TRANSPOSE_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_TRANSPOSE_H_

#include "kernel/pyboost/op_register.h"
#include "mindspore/core/ops/view/transpose_strides_calc.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class BACKEND_EXPORT Transpose : public pyboost::Op {
 public:
  Transpose() = default;
  ~Transpose() = default;

  void CastInput() override;
  virtual tensor::TensorPtr Call(const tensor::TensorPtr &input, const ValueTuplePtr &input_perm);
  void PyboostProcessView(const tensor::TensorPtr &input, const std::vector<int64_t> &input_perm);
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_TRANSPOSE_H_
