/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_KERNELS_IMAGE_RANDOM_COLOR_ADJUST_OP_H_
#define DATASET_KERNELS_IMAGE_RANDOM_COLOR_ADJUST_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomColorAdjustOp : public TensorOp {
 public:
  static const uint32_t kDefSeed;

  // Constructor for RandomColorAdjustOp.
  // @param s_bright_factor brightness change range start value.
  // @param e_bright_factor brightness change range end value.
  // @param s_contrast_factor contrast change range start value.
  // @param e_contrast_factor contrast change range start value.
  // @param s_saturation_factor saturation change range end value.
  // @param e_saturation_factor saturation change range end value.
  // @param s_hue_factor hue change factor start value, this should be greater than  -0.5.
  // @param e_hue_factor hue change factor start value, this should be less than  0.5.
  // @param seed optional seed to pass in to the constructor.
  // @details the randomly chosen degree is uniformly distributed.
  RandomColorAdjustOp(float s_bright_factor, float e_bright_factor, float s_contrast_factor, float e_contrast_factor,
                      float s_saturation_factor, float e_saturation_factor, float s_hue_factor, float e_hue_factor);

  ~RandomColorAdjustOp() override = default;

  // Print function for RandomJitter.
  // @param out output stream to print to.
  void Print(std::ostream &out) const override { out << "RandomColorAdjustOp: "; }

  // Overrides the base class compute function.
  // Calls multiple transform functions in ImageUtils, this function takes an input tensor.
  // and transforms its data using openCV, the output memory is manipulated to contain the result.
  // @return Status - The error code return.
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  std::mt19937 rnd_;
  float bright_factor_start_;
  float bright_factor_end_;
  float contrast_factor_start_;
  float contrast_factor_end_;
  float saturation_factor_start_;
  float saturation_factor_end_;
  float hue_factor_start_;
  float hue_factor_end_;
  // Compare two floating point variables. Return true if they are same / very close.
  inline bool CmpFloat(const float &a, const float &b, float epsilon = 0.0000000001f) const {
    return (std::fabs(a - b) < epsilon);
  }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_RANDOM_COLOR_ADJUST_OP_H_
