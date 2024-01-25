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
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_resized_crop.h"

#include <vector>

#include "minddata/dataset/kernels/data/data_utils.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/dvpp/utils/dvpp_image_utils.h"
#include "minddata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "minddata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const int32_t DvppResizedCropOp::kDefWidth = 0;
const InterpolationMode DvppResizedCropOp::kDefInterpolation = InterpolationMode::kLinear;

Status DvppResizedCropOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                                  std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);
  // the input should be NHWC, N is 1.
  const auto kNHWCImageRank = 4;
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->GetShape().Rank() == kNHWCImageRank,
    "DvppResizedCrop: the input tensor is not HW, HWC or 1HWC, but got: " + std::to_string(input->GetShape().Rank()));
  // the channel should be 3 or 1
  const auto kChannelIndexNHWC = 3;
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetShape().AsVector()[kChannelIndexNHWC] == kMinImageChannel ||
                                 input->GetShape().AsVector()[kChannelIndexNHWC] == kDefaultImageChannel,
                               "DvppResizedCrop: the channel of the input is not 1 or 3, but got: " +
                                 std::to_string(input->GetShape().AsVector()[kChannelIndexNHWC]));

  // the type should be uint8 or float
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetType() == DataType::DE_UINT8 || input->GetType() == DataType::DE_FLOAT32,
                               "DvppResizedCrop: the type of the input is not uint8 or float.");
  const auto kWidthIndexNHWC = 2;
  const auto kHeightIndexNHWC = 1;
  std::vector<dsize_t> size = {input->GetShape().AsVector()[kHeightIndexNHWC],
                               input->GetShape().AsVector()[kWidthIndexNHWC]};
  int32_t input_h = size[kHeightIndex];
  int32_t input_w = size[kWidthIndex];
  int32_t output_h;
  int32_t output_w;
  int32_t size1 = size_[kHeightIndex];
  int32_t size2 = size_[kWidthIndex];

  // crop check
  CHECK_FAIL_RETURN_UNEXPECTED(top_ + height_ <= input_h,
                               "DvppResizedCrop: the sum of top and height: " + std::to_string(top_ + height_) +
                                 " exceeds image height: " + std::to_string(input_h));
  CHECK_FAIL_RETURN_UNEXPECTED(left_ + width_ <= input_w,
                               "DvppResizedCrop: the sum of left and width: " + std::to_string(left_ + width_) +
                                 " exceeds image width: " + std::to_string(input_w));

  if (size2 == 0) {
    if (input_h < input_w) {
      CHECK_FAIL_RETURN_UNEXPECTED(input_h != 0, "DvppResizedCrop: the input height cannot be 0.");
      output_h = size1;
      output_w = static_cast<int>(std::floor(static_cast<float>(input_w) / input_h * output_h));
    } else {
      CHECK_FAIL_RETURN_UNEXPECTED(input_w != 0, "DvppResizedCrop: the input width cannot be 0.");
      output_w = size1;
      output_h = static_cast<int>(std::floor(static_cast<float>(input_h) / input_w * output_w));
    }
  } else {
    output_h = size1;
    output_w = size2;
  }
  if (input_h == output_h && input_w == output_w) {
    *output = input;
    return Status::OK();
  }

  // verify InterpolationMode
  CHECK_FAIL_RETURN_UNEXPECTED(GetDVPPInterpolationMode(interpolation_) != kInvalidInterpolationMode,
                               "The current InterpolationMode is not supported by DVPP. It is " +
                                 std::to_string(static_cast<int>(interpolation_)));

  APP_ERROR ret = AclAdapter::GetInstance().DvppResizedCrop(input, output, top_, left_, height_, width_, output_h,
                                                            output_w, interpolation_);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppResizedCrop: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }

  return Status::OK();
}

Status DvppResizedCropOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  int32_t outputH = size_[kHeightIndex];
  int32_t outputW = size_[kWidthIndex];
  if (inputs[0].Rank() < kMinImageRank) {
    std::string err_msg =
      "DvppResizedCrop: input tensor should have at least 2 dimensions, but got: " + std::to_string(inputs[0].Rank());
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else if (inputs[0].Rank() == kMinImageRank) {
    TensorShape out = TensorShape{outputH, outputW};
    (void)outputs.emplace_back(out);
  } else if (inputs[0].Rank() >= kDefaultImageRank) {
    auto out_shape = ComputeOutputShape(inputs[0], outputH, outputW);
    (void)outputs.emplace_back(out_shape);
  }
  return Status::OK();
}

TensorShape DvppResizedCropOp::ComputeOutputShape(const TensorShape &input, int32_t output_h, int32_t output_w) {
  const int kHeightIndex = -3;
  const int kWidthIndex = -2;
  auto out_shape_vec = input.AsVector();
  auto size = out_shape_vec.size();
  out_shape_vec[size + kHeightIndex] = output_h;
  out_shape_vec[size + kWidthIndex] = output_w;
  TensorShape out = TensorShape(out_shape_vec);
  return out;
}
}  // namespace dataset
}  // namespace mindspore
