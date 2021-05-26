/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_APPLY_OP_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_APPLY_OP_

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/data/compose_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
class RandomApplyOp : public TensorOp {
 public:
  /// constructor
  /// \param[in] ops the list of TensorOps to apply with prob likelihood
  /// \param[in] prob probability whether the list of TensorOps will be applied
  explicit RandomApplyOp(const std::vector<std::shared_ptr<TensorOp>> &ops, double prob);

  /// default destructor
  ~RandomApplyOp() = default;

  /// return the number of inputs the first tensorOp in compose takes
  /// \return number of input tensors
  uint32_t NumInput() override { return compose_->NumInput(); }

  /// return the number of outputs
  /// \return number of output tensors
  uint32_t NumOutput() override;

  /// return output shape if randomApply won't affect the output shape, otherwise return unknown shape
  /// \param[in] inputs
  /// \param[out] outputs
  /// \return  Status code
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  /// return output type if randomApply won't affect the output type, otherwise return unknown type
  /// \param[in] inputs
  /// \param[out] outputs
  /// \return Status code
  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  /// \param[in] input
  /// \param[out] output
  /// \return Status code
  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomApplyOp; }

 private:
  double prob_;
  std::shared_ptr<TensorOp> compose_;
  std::mt19937 gen_;  // mersenne_twister_engine
  std::uniform_real_distribution<double> rand_double_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_APPLY_OP_
