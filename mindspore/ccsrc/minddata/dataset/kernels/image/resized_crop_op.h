/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZED_CROP_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZED_CROP_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class ResizedCropOp : public TensorOp {
 public:
  ResizedCropOp(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<int32_t> &size,
                InterpolationMode interpolation)
      : top_(top), left_(left), height_(height), width_(width), size_(size), interpolation_(interpolation) {}

  ~ResizedCropOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": " << top_ << " " << left_ << " " << height_ << " " << width_;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kResizedCropOp; }

 protected:
  int32_t top_;
  int32_t left_;
  int32_t height_;
  int32_t width_;
  const std::vector<int32_t> size_;
  InterpolationMode interpolation_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZED_CROP_OP_H_
