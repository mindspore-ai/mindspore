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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_CHOICE_OP_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_CHOICE_OP_

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/kernels/data/compose_op.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
class RandomChoiceOp : public TensorOp {
 public:
  /// constructor
  /// \param[in] ops list of TensorOps to randomly choose 1 from
  explicit RandomChoiceOp(const std::vector<std::shared_ptr<TensorOp>> &ops);

  /// default destructor
  ~RandomChoiceOp() = default;

  /// return the number of inputs. All op in ops_ should have the same number of inputs
  /// \return number of input tensors
  uint32_t NumInput() override;

  /// return the number of outputs. All op in ops_ should have the same number of outputs
  /// \return number of input tensors
  uint32_t NumOutput() override;

  /// return output shape if all ops in ops_ return the same shape, otherwise return unknown shape
  /// \param[in] inputs
  /// \param[out] outputs
  /// \return  Status code
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  /// return output type if all ops in ops_ return the same type, otherwise return unknown type
  /// \param[in] inputs
  /// \param[out] outputs
  /// \return Status code
  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  /// \param[in] input
  /// \param[out] output
  /// \return Status code
  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomChoiceOp; }

 private:
  std::vector<std::shared_ptr<TensorOp>> ops_;
  std::mt19937 gen_;  // mersenne_twister_engine
  std::uniform_int_distribution<size_t> rand_int_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_CHOICE_OP_
