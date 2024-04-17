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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomCropOp : public RandomTensorOp {
 public:
  RandomCropOp(int32_t crop_height, int32_t crop_width, int32_t pad_top, int32_t pad_bottom, int32_t pad_left,
               int32_t pad_right, bool pad_if_needed, BorderType padding_mode, uint8_t fill_r, uint8_t fill_g,
               uint8_t fill_b);

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
  void GenRandomXY(int32_t *x, int32_t *y, int32_t padded_image_w, int32_t padded_image_h);

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kRandomCropOp; }

  uint32_t NumInput() override { return 1; }

  uint32_t NumOutput() override { return 1; }

 protected:
  int32_t crop_height_ = 0;
  int32_t crop_width_ = 0;

 private:
  Status RandomCropImg(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t *x, int32_t *y,
                       int32_t index);

  Status ConstructShape(const TensorShape &in_shape, std::shared_ptr<TensorShape> *out_shape) const;

  int32_t pad_top_ = 0;
  int32_t pad_bottom_ = 0;
  int32_t pad_left_ = 0;
  int32_t pad_right_ = 0;
  bool pad_if_needed_ = false;
  BorderType border_type_;
  uint8_t fill_r_ = 0;
  uint8_t fill_g_ = 0;
  uint8_t fill_b_ = 0;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_OP_H_
