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
#ifndef DATASET_KERNELS_IMAGE_RANDOM_CROP_OP_H_
#define DATASET_KERNELS_IMAGE_RANDOM_CROP_OP_H_

#include <memory>
#include <random>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomCropOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefPadTop;
  static const int32_t kDefPadBottom;
  static const int32_t kDefPadLeft;
  static const int32_t kDefPadRight;
  static const BorderType kDefBorderType;
  static const bool kDefPadIfNeeded;
  static const uint8_t kDefFillR;
  static const uint8_t kDefFillG;
  static const uint8_t kDefFillB;

  RandomCropOp(int32_t crop_height, int32_t crop_width, int32_t pad_top = kDefPadTop,
               int32_t pad_bottom = kDefPadBottom, int32_t pad_left = kDefPadLeft, int32_t pad_right = kDefPadRight,
               BorderType border_types = kDefBorderType, bool pad_if_needed = kDefPadIfNeeded,
               uint8_t fill_r = kDefFillR, uint8_t fill_g = kDefFillG, uint8_t fill_b = kDefFillB);

  ~RandomCropOp() override = default;

  void Print(std::ostream &out) const override { out << "RandomCropOp: " << crop_height_ << " " << crop_width_; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status ImagePadding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *pad_image, int32_t *t_pad_top,
                      int32_t *t_pad_bottom, int32_t *t_pad_left, int32_t *t_pad_right, int32_t *padded_image_w,
                      int32_t *padded_image_h, bool *crop_further);

  void GenRandomXY(int *x, int *y, int32_t *padded_image_w, int32_t *padded_image_h);

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

 protected:
  int32_t crop_height_ = 0;
  int32_t crop_width_ = 0;

 private:
  int32_t pad_top_ = 0;
  int32_t pad_bottom_ = 0;
  int32_t pad_left_ = 0;
  int32_t pad_right_ = 0;
  bool pad_if_needed_ = false;
  BorderType border_type_;
  uint8_t fill_r_ = 0;
  uint8_t fill_g_ = 0;
  uint8_t fill_b_ = 0;
  std::mt19937 rnd_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_RANDOM_CROP_OP_H_
