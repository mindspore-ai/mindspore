/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_OP_H_

#include <memory>
#include <random>
#include <vector>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

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
               bool pad_if_needed = kDefPadIfNeeded, BorderType padding_mode = kDefBorderType,
               uint8_t fill_r = kDefFillR, uint8_t fill_g = kDefFillG, uint8_t fill_b = kDefFillB);

  RandomCropOp(const RandomCropOp &rhs) = default;

  RandomCropOp(RandomCropOp &&rhs) = default;

  ~RandomCropOp() override = default;

  void Print(std::ostream &out) const override { out << Name() << ": " << crop_height_ << " " << crop_width_; }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  // Function breaks out the compute function's image padding functionality and makes available to other Ops
  // Using this class as a base - re-structured to allow for RandomCropWithBBox Augmentation Op
  // @param input: Input is the original Image
  // @param pad_image: Pointer to new Padded image
  // @param t_pad_top: Total Top Padding - Based on input and value calculated in function if required
  // @param t_pad_bottom: Total bottom Padding - Based on input and value calculated in function if required
  // @param t_pad_left: Total left Padding - Based on input and value calculated in function if required
  // @param t_pad_right: Total right Padding - Based on input and value calculated in function if required
  // @param padded_image_w: Final Width of the 'pad_image'
  // @param padded_image_h: Final Height of the 'pad_image'
  // @param crop_further: Whether image required cropping after padding - False if new padded image matches required
  // dimensions
  Status ImagePadding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *pad_image, int32_t *t_pad_top,
                      int32_t *t_pad_bottom, int32_t *t_pad_left, int32_t *t_pad_right, int32_t *padded_image_w,
                      int32_t *padded_image_h, bool *crop_further);

  // Function breaks X,Y generation functionality out of original compute function and makes available to other Ops
  void GenRandomXY(int *x, int *y, const int32_t &padded_image_w, const int32_t &padded_image_h);

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kRandomCropOp; }

  uint32_t NumInput() override { return -1; }

  uint32_t NumOutput() override { return -1; }

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

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_OP_H_
