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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_ROTATION_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_ROTATION_OP_H_

#include <memory>
#include <random>
#include <vector>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
class RandomRotationOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const std::vector<float> kDefCenter;
  static const InterpolationMode kDefInterpolation;
  static const bool kDefExpand;
  static const uint8_t kDefFillR;
  static const uint8_t kDefFillG;
  static const uint8_t kDefFillB;

  // Constructor for RandomRotationOp
  // @param startDegree starting range for random degree
  // @param endDegree ending range for random degree
  // @param interpolation DE interpolation mode for rotation
  // @param expand option for the output image shape to change
  // @param center coordinate for center of image rotation
  // @param fill_r R value for the color to pad with
  // @param fill_g G value for the color to pad with
  // @param fill_b B value for the color to pad with
  // @details the randomly chosen degree is uniformly distributed
  // @details the output shape, if changed, will contain the entire rotated image
  // @note maybe using unsigned long int isn't the best here according to our coding rules
  RandomRotationOp(float start_degree, float end_degree, InterpolationMode resample = kDefInterpolation,
                   bool expand = kDefExpand, std::vector<float> center = kDefCenter, uint8_t fill_r = kDefFillR,
                   uint8_t fill_g = kDefFillG, uint8_t fill_b = kDefFillB);

  ~RandomRotationOp() override = default;

  // Overrides the base class compute function
  // Calls the rotate function in ImageUtils, this function takes an input tensor
  // and transforms its data using openCV, the output memory is manipulated to contain the result
  // @return Status The status code returned
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kRandomRotationOp; }

 private:
  float degree_start_;
  float degree_end_;
  std::vector<float> center_;
  InterpolationMode interpolation_;
  bool expand_;
  uint8_t fill_r_;
  uint8_t fill_g_;
  uint8_t fill_b_;
  std::uniform_real_distribution<float> distribution_{-1.0, 1.0};
  std::mt19937 rnd_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_ROTATION_OP_H_
