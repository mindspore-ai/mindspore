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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_SELECT_SUBPOLICY_OP_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_SELECT_SUBPOLICY_OP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
using Subpolicy = std::vector<std::pair<std::shared_ptr<TensorOp>, double>>;

class RandomSelectSubpolicyOp : public RandomTensorOp {
 public:
  /// constructor
  /// \param[in] policy policy to choose subpolicy from
  explicit RandomSelectSubpolicyOp(const std::vector<Subpolicy> &policy);

  /// destructor
  ~RandomSelectSubpolicyOp() override = default;

  /// return number of input tensors
  /// \return number of inputs if all ops in policy have the same NumInput, otherwise return 0
  uint32_t NumInput() override;

  /// return number of output tensors
  /// \return number of outputs if all ops in policy have the same NumOutput, otherwise return 0
  uint32_t NumOutput() override;

  /// return unknown shapes
  /// \param[in] inputs
  /// \param[out] outputs
  /// \return Status Code
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  /// return output type if all ops in policy return the same type, otherwise return unknown type
  /// \param[in] inputs
  /// \param[out] outputs
  /// \return Status Code
  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  /// \param[in] input
  /// \param[out] output
  /// \return Status code
  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomSelectSubpolicyOp; }

 private:
  std::vector<Subpolicy> policy_;
  std::uniform_int_distribution<size_t> rand_int_;
  std::uniform_real_distribution<double> rand_double_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_SELECT_SUBPOLICY_OP_
