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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_SHARPNESS_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_SHARPNESS_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomSharpnessOp : public RandomTensorOp {
 public:
  /// Adjust the sharpness of the input image by a random degree within the given range.
  /// \@param[in] start_degree A float indicating the beginning of the range.
  /// \@param[in] end_degree A float indicating the end of the range.
  explicit RandomSharpnessOp(float start_degree, float end_degree);

  ~RandomSharpnessOp() override = default;

  void Print(std::ostream &out) const override { out << Name(); }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRandomSharpnessOp; }

 protected:
  float start_degree_;
  float end_degree_;
  std::uniform_real_distribution<float> distribution_{-1.0, 1.0};
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_SHARPNESS_OP_H_
