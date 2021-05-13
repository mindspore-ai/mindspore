/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_TO_NUMBER_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_TO_NUMBER_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class ToNumberOp : public TensorOp {
 public:
  // Constructor of ToNumberOp
  // @param const DataType &data_type - the type to convert string inputs to.
  explicit ToNumberOp(const DataType &data_type);

  // Constructor of ToNumberOp
  // @param const std::string &data_type - the type in string form to convert string inputs to.
  explicit ToNumberOp(const std::string &data_type);

  ~ToNumberOp() override = default;

  // Perform numeric conversion on each string in each tensor.
  // @param const std::shared_ptr<Tensor> &input
  // @param std::shared_ptr<Tensor> *output
  // @return error code
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  // For each input shape, find the output shape
  // @param std::vector<TensorShape> &inputs - shape of input tensors
  // @param std::vector<TensorShape> &outputs - shape of output tensors
  // @return error code
  Status OutputShape(const std::vector<TensorShape> &input_shapes, std::vector<TensorShape> &output_shapes) override;

  // print arg for debugging
  // @param std::ostream &out
  void Print(std::ostream &out) const override;

  std::string Name() const override { return kToNumberOp; }

 private:
  template <typename T>
  Status ToSignedIntegral(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) const;

  template <typename T>
  Status ToUnsignedIntegral(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) const;

  Status ToFloat16(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) const;

  Status ToFloat(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) const;

  Status ToDouble(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) const;

  DataType cast_to_type_;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_TO_NUMBER_OP_H_
