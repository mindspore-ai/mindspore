
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
#ifndef DATASET_KERNELS_IMAGE_SOFT_DVPP_DECODE_RANDOM_CROP_RESIZE_JPEG_OP_H_
#define DATASET_KERNELS_IMAGE_SOFT_DVPP_DECODE_RANDOM_CROP_RESIZE_JPEG_OP_H_

#include <memory>
#include <random>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/external_soft_dp.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class SoftDvppDecodeRandomCropResizeJpegOp : public TensorOp {
 public:
  static const float kDefScaleLb;
  static const float kDefScaleUb;
  static const float kDefAspectLb;
  static const float kDefAspectUb;
  static const InterpolationMode kDefInterpolation;
  static const int32_t kDefMaxIter;

  SoftDvppDecodeRandomCropResizeJpegOp(int32_t target_height, int32_t target_width, float scale_lb = kDefScaleLb,
                                       float scale_ub = kDefScaleUb, float aspect_lb = kDefAspectLb,
                                       float aspect_ub = kDefAspectUb, int32_t max_attempts = kDefMaxIter);

  /// \brief Destructor
  ~SoftDvppDecodeRandomCropResizeJpegOp() = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kSoftDvppDecodeRandomCropResizeJpegOp; }

 protected:
  Status GetCropInfo(const std::shared_ptr<Tensor> &input, SoftDpCropInfo *crop_info);

  int32_t target_height_;
  int32_t target_width_;
  float scale_lb_;
  float scale_ub_;
  float aspect_lb_;
  float aspect_ub_;
  int32_t max_attempts_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_SOFT_DVPP_DECODE_RANDOM_CROP_RESIZE_JPEG_OP_H_
