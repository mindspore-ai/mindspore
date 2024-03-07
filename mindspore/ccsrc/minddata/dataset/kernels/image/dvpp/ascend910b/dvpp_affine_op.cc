/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_affine_op.h"

#include <algorithm>
#include <vector>

#include "minddata/dataset/kernels/data/data_utils.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "minddata/dataset/kernels/image/dvpp/utils/dvpp_image_utils.h"
#include "minddata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
constexpr int64_t h_lb = 4;      // height lower bound
constexpr int64_t h_ub = 32768;  // height upper bound
constexpr int64_t w_lb = 6;      // width lower bound
constexpr int64_t w_ub = 32768;  // width upper bound

DvppAffineOp::DvppAffineOp(float degrees, const std::vector<float> &translation, float scale,
                           const std::vector<float> &shear, InterpolationMode interpolation,
                           const std::vector<uint8_t> &fill_value)
    : degrees_(degrees),
      translation_(translation),
      scale_(scale),
      shear_(shear),
      interpolation_(interpolation),
      fill_value_(fill_value) {}

Status DvppAffineOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                             std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);

  // the input should be NHWC, N is 1.
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->GetShape().Rank() == kNHWCImageRank,
    "DvppAffine: the input tensor is not HW, HWC or 1HWC, but got: " + std::to_string(input->GetShape().Rank()));

  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour) {
    auto error = "DvppAffine: Invalid interpolation mode, only support BILINEAR and NEAREST";
    RETURN_STATUS_UNEXPECTED(error);
  }

  // Dvpp Limit
  int64_t input_h = input->GetShape()[kHeightIndexNHWC];
  int64_t input_w = input->GetShape()[kWidthIndexNHWC];
  RETURN_IF_NOT_OK(CheckDvppLimit(input_h, input_w, h_lb, w_lb, h_ub, w_ub, kDvppAffineOp));

  // run dvpp
  uint32_t interpolation_mode = GetDVPPInterpolationMode(interpolation_);
  uint32_t padding_mode = static_cast<uint32_t>(BorderType::kConstant);

  float_t translation_x = translation_[0] * static_cast<float>(input_w);
  float_t translation_y = translation_[1] * static_cast<float>(input_h);
  std::vector<float> translation_xy{translation_x, translation_y};

  // fake tensor here to reuse the GetAffineMatrix
  std::shared_ptr<Tensor> fake_tensor = std::make_shared<Tensor>(TensorShape({input_h, input_w}), DataType("uint8"));
  std::vector<float> matrix;
  RETURN_IF_NOT_OK(GetAffineMatrix(fake_tensor, &matrix, degrees_, translation_xy, scale_, shear_));

  std::vector<float> fill;
  if (fill_value_.size() == 1) {
    fill_value_ = {fill_value_[0], fill_value_[0], fill_value_[0]};
  }
  (void)std::transform(fill_value_.begin(), fill_value_.end(), std::back_inserter(fill),
                       [](const auto &v) { return static_cast<float>(v); });

  APP_ERROR ret = AclAdapter::GetInstance().DvppAffine(input, output, matrix, interpolation_mode, padding_mode, fill);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppAffine: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppAffineOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  return Status::OK();
}

Status DvppAffineOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
