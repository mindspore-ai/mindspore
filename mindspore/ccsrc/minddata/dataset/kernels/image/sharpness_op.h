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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_SHARPNESS_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_SHARPNESS_OP_H_

#include <memory>
#include <vector>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class SharpnessOp : public TensorOp {
 public:
  /// Default values, also used by bindings.cc
  static const float kDefAlpha;

  /// This class can be used to adjust the sharpness of an image.
  /// \@param[in] alpha A float indicating the enhancement factor.
  /// a factor of 0.0 gives a blurred image, a factor of 1.0 gives the
  /// original image, and a factor of 2.0 gives a sharpened image.

  explicit SharpnessOp(const float alpha = kDefAlpha) : alpha_(alpha) {}

  ~SharpnessOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kSharpnessOp; }

 protected:
  float alpha_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_SHARPNESS_OP_H_
