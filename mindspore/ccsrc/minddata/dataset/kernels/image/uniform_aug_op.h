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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_UNIFORM_AUG_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_UNIFORM_AUG_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class UniformAugOp : public TensorOp {
 public:
  // Default number of Operations to be applied
  static const int kDefNumOps;

  // Constructor for UniformAugOp
  // @param std::vector<std::shared_ptr<TensorOp>> op_list: list of candidate C++ operations
  // @param int32_t num_ops: number of augemtation operations to applied
  UniformAugOp(std::vector<std::shared_ptr<TensorOp>> op_list, int32_t num_ops);

  // Destructor
  ~UniformAugOp() override = default;

  void Print(std::ostream &out) const override { out << Name() << ":: number of ops " << num_ops_; }

  // Overrides the base class compute function
  // @return Status The status code returned
  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kUniformAugOp; }

 private:
  std::vector<std::shared_ptr<TensorOp>> tensor_op_list_;
  int32_t num_ops_;
  std::mt19937 rnd_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_UNIFORM_AUG_OP_H_
