/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_TO_VECTORS_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_TO_VECTORS_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/text/vectors.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class ToVectorsOp : public TensorOp {
 public:
  /// \brief Constructor.
  /// \param[in] vectors Vectors used to lookup tokens.
  /// \param[in] unk_init Vector used to initialize OOV token.
  /// \param[in] lower_case_backup Whether to look up the token in the lower case.
  ToVectorsOp(const std::shared_ptr<Vectors> &vectors, const std::vector<float> &unk_init, bool lower_case_backup);

  /// \brief Destructor.
  ~ToVectorsOp() = default;

  /// \brief Perform actual ToVectors on each tensor.
  /// \param[in] input Input tensor.
  /// \param[in] output Output tensor.
  /// \return[out] Status code.
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  /// \param[in] inputs DataType of input tensor.
  /// \param[in] outputs DataType of output tensor.
  /// \return[out] Status code.
  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  /// \brief Get Op name.
  std::string Name() const override { return kToVectorsOp; }

 private:
  std::shared_ptr<Vectors> vectors_;
  std::vector<float> unk_init_;
  bool lower_case_backup_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_TO_VECTORS_OP_H_
