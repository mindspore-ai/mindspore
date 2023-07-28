/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_C_FUNC_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_C_FUNC_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class CFuncOp : public TensorOp {
 public:
  explicit CFuncOp(const std::function<TensorRow(TensorRow)> &func) : c_func_ptr_(func) {}

  ~CFuncOp() override = default;

  uint32_t NumInput() override { return 0; }

  uint32_t NumOutput() override { return 0; }

  // Calls c_func_ptr and returns the result
  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kCFuncOp; }

 private:
  std::function<TensorRow(TensorRow)> c_func_ptr_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_C_FUNC_OP_H_
