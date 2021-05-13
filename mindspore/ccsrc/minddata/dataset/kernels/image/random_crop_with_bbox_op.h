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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_WITH_BBOX_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_WITH_BBOX_OP_H_

#include <memory>
#include <vector>
#include <string>

#include "minddata/dataset/kernels/image/random_crop_op.h"

namespace mindspore {
namespace dataset {
class RandomCropWithBBoxOp : public RandomCropOp {
 public:
  //  Constructor for RandomCropWithBBoxOp, with default value and passing to base class constructor
  RandomCropWithBBoxOp(int32_t crop_height, int32_t crop_width, int32_t pad_top = kDefPadTop,
                       int32_t pad_bottom = kDefPadBottom, int32_t pad_left = kDefPadLeft,
                       int32_t pad_right = kDefPadRight, bool pad_if_needed = kDefPadIfNeeded,
                       BorderType padding_mode = kDefBorderType, uint8_t fill_r = kDefFillR, uint8_t fill_g = kDefFillG,
                       uint8_t fill_b = kDefFillB)
      : RandomCropOp(crop_height, crop_width, pad_top, pad_bottom, pad_left, pad_right, pad_if_needed, padding_mode,
                     fill_r, fill_g, fill_b) {}

  ~RandomCropWithBBoxOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": " << RandomCropOp::crop_height_ << " " << RandomCropOp::crop_width_;
  }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomCropWithBBoxOp; }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_WITH_BBOX_OP_H_
