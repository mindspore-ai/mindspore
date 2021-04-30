/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/random_crop_decode_resize_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_crop_decode_resize_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {

namespace vision {
#ifndef ENABLE_ANDROID

// RandomCropDecodeResizeOperation
RandomCropDecodeResizeOperation::RandomCropDecodeResizeOperation(std::vector<int32_t> size, std::vector<float> scale,
                                                                 std::vector<float> ratio,
                                                                 InterpolationMode interpolation, int32_t max_attempts)
    : RandomResizedCropOperation(size, scale, ratio, interpolation, max_attempts) {}

RandomCropDecodeResizeOperation::~RandomCropDecodeResizeOperation() = default;

std::string RandomCropDecodeResizeOperation::Name() const { return kRandomCropDecodeResizeOperation; }

std::shared_ptr<TensorOp> RandomCropDecodeResizeOperation::Build() {
  int32_t crop_height = size_[0];
  int32_t crop_width = size_[0];

  // User has specified the crop_width value.
  if (size_.size() == 2) {
    crop_width = size_[1];
  }

  float scale_lower_bound = scale_[0];
  float scale_upper_bound = scale_[1];

  float aspect_lower_bound = ratio_[0];
  float aspect_upper_bound = ratio_[1];

  auto tensor_op =
    std::make_shared<RandomCropDecodeResizeOp>(crop_height, crop_width, scale_lower_bound, scale_upper_bound,
                                               aspect_lower_bound, aspect_upper_bound, interpolation_, max_attempts_);
  return tensor_op;
}

RandomCropDecodeResizeOperation::RandomCropDecodeResizeOperation(const RandomResizedCropOperation &base)
    : RandomResizedCropOperation(base) {}

Status RandomCropDecodeResizeOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = size_;
  args["scale"] = scale_;
  args["ratio"] = ratio_;
  args["interpolation"] = interpolation_;
  args["max_attempts"] = max_attempts_;
  *out_json = args;
  return Status::OK();
}

#endif

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
