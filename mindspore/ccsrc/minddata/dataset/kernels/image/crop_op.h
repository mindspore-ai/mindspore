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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_CROP_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_CROP_OP_H_

#include <memory>
#include <vector>
#include <string>

#include "minddata/dataset/core/tensor.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class CropOp : public TensorOp {
 public:
  /// \brief Constructor to Crop Op
  /// \param[in] y - the vertical starting coordinate
  /// \param[in] x - the horizontal starting coordinate
  /// \param[in] height - the height of the crop box
  /// \param[in] width - the width of the crop box
  explicit CropOp(int32_t y, int32_t x, int32_t height, int32_t width) : y_(y), x_(x), height_(height), width_(width) {}

  CropOp(const CropOp &rhs) = default;

  CropOp(CropOp &&rhs) = default;

  ~CropOp() override = default;

  void Print(std::ostream &out) const override {
    out << "CropOp y: " << y_ << " x: " << x_ << " h: " << height_ << " w: " << width_;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kCropOp; }

 protected:
  int32_t y_;
  int32_t x_;
  int32_t height_;
  int32_t width_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_CROP_OP_H_
