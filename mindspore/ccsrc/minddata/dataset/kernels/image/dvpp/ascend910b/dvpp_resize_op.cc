/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_resize_op.h"

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
const int32_t DvppResizeOp::kDefWidth = 0;
const InterpolationMode DvppResizeOp::kDefInterpolation = InterpolationMode::kLinear;

Status DvppResizeOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                             std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);
  // the input should be NHWC, N is 1.
  const auto kNHWCImageRank = 4;
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetShape().Rank() == kNHWCImageRank,
                               "DvppResize: the input tensor is not HW, HWC or 1HWC.");
  // the channel should be 3 or 1
  const auto kChannelIndexNHWC = 3;
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetShape().AsVector()[kChannelIndexNHWC] == 1 ||
                                 input->GetShape().AsVector()[kChannelIndexNHWC] == kDefaultImageChannel,
                               "DvppResize: the channel of the input is not 1 or 3.");

  // the type should be uint8 or float
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetType() == DataType::DE_UINT8 || input->GetType() == DataType::DE_FLOAT32,
                               "DvppResize: the type of the input is not uint8 or float.");
  const auto kWidthIndexNHWC = 2;
  std::vector<dsize_t> size = {input->GetShape().AsVector()[1], input->GetShape().AsVector()[kWidthIndexNHWC]};
  int32_t input_h = size[kHeightIndex];
  int32_t input_w = size[kWidthIndex];
  int32_t output_h;
  int32_t output_w;
  if (size2_ == 0) {
    if (input_h < input_w) {
      CHECK_FAIL_RETURN_UNEXPECTED(input_h != 0, "DvppResize: the input height cannot be 0.");
      output_h = size1_;
      output_w = static_cast<int>(std::floor(static_cast<float>(input_w) / input_h * output_h));
    } else {
      CHECK_FAIL_RETURN_UNEXPECTED(input_w != 0, "DvppResize: the input width cannot be 0.");
      output_w = size1_;
      output_h = static_cast<int>(std::floor(static_cast<float>(input_h) / input_w * output_w));
    }
  } else {
    output_h = size1_;
    output_w = size2_;
  }
  if (input_h == output_h && input_w == output_w) {
    *output = input;
    return Status::OK();
  }

  // verify InterpolationMode
  CHECK_FAIL_RETURN_UNEXPECTED(GetDVPPInterpolationMode(interpolation_) != kInvalidInterpolationMode,
                               "The current InterpolationMode is not supported by DVPP. It is " +
                                 std::to_string(static_cast<int>(interpolation_)));

  APP_ERROR ret = AclAdapter::GetInstance().DvppResize(input, output, output_h, output_w, 0, 0, interpolation_);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppResize: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }

  return Status::OK();
}

Status DvppResizeOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  int32_t outputH = -1;
  int32_t outputW = -1;
  // if size2_ == 0, then we cannot know the shape. We need the input image to find the output shape --> set it to
  // <-1,-1[,3]>
  if (size2_ != 0) {
    outputH = size1_;
    outputW = size2_;
  }
  if (inputs[0].Rank() < kMinImageRank) {
    std::string err_msg =
      "DvppResize: input tensor should have at least 2 dimensions, but got: " + std::to_string(inputs[0].Rank());
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

TensorShape DvppResizeOp::ComputeOutputShape(const TensorShape &input, int32_t output_h, int32_t output_w) {
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
