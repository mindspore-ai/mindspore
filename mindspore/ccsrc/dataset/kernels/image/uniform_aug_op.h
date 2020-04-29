/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef DATASET_KERNELS_IMAGE_UNIFORM_AUG_OP_H_
#define DATASET_KERNELS_IMAGE_UNIFORM_AUG_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"
#include "dataset/kernels/py_func_op.h"

#include "pybind11/stl.h"

namespace mindspore {
namespace dataset {
class UniformAugOp : public TensorOp {
 public:
  // Default number of Operations to be applied
  static const int kDefNumOps;

  // Constructor for UniformAugOp
  // @param list op_list: list of candidate C++ operations
  // @param list num_ops: number of augemtation operations to applied
  UniformAugOp(py::list op_list, int32_t num_ops);

  ~UniformAugOp() override = default;

  void Print(std::ostream &out) const override { out << "UniformAugOp:: number of ops " << num_ops_; }

  // Overrides the base class compute function
  // @return Status - The error code return
  Status Compute(const std::vector<std::shared_ptr<Tensor>> &input,
                 std::vector<std::shared_ptr<Tensor>> *output) override;

 private:
  int32_t num_ops_;
  std::vector<std::shared_ptr<TensorOp>> tensor_op_list_;
  std::mt19937 rnd_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_UNIFORM_AUG_OP_H_
