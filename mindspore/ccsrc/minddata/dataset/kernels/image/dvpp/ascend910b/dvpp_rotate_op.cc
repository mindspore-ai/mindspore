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
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_rotate_op.h"

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
constexpr int64_t h_lb = 4;       // height lower bound
constexpr int64_t h_ub = 32768;   // height upper bound
constexpr int64_t w_lb = 6;       // width lower bound
constexpr int64_t w_ub = 32768;   // width upper bound
constexpr int64_t pad_ub = 2048;  // max padding number

DvppRotateOp::DvppRotateOp(float degrees, InterpolationMode resample, bool expand, const std::vector<float> &center,
                           uint8_t fill_r, uint8_t fill_g, uint8_t fill_b)
    : degrees_(degrees),
      resample_(resample),
      expand_(expand),
      center_(center),
      fill_r_(fill_r),
      fill_g_(fill_g),
      fill_b_(fill_b) {}

Status DvppRotateOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                             std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);

  // the input should be NHWC, N is 1.
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->GetShape().Rank() == kNHWCImageRank,
    "DvppRotate: the input tensor is not HW, HWC or 1HWC, but got: " + std::to_string(input->GetShape().Rank()));

  std::vector<float> fill = {static_cast<float>(fill_b_), static_cast<float>(fill_g_), static_cast<float>(fill_r_)};

  // verify InterpolationMode
  CHECK_FAIL_RETURN_UNEXPECTED(GetDVPPRotateMode(resample_) != kInvalidRotateMode,
                               "DvppRotate: Invalid interpolation mode, only support BILINEAR and NEAREST.");

  // Dvpp Limit
  int64_t input_h = input->GetShape()[kHeightIndexNHWC];
  int64_t input_w = input->GetShape()[kWidthIndexNHWC];
  RETURN_IF_NOT_OK(CheckDvppLimit(input_h, input_w, h_lb, w_lb, h_ub, w_ub, kDvppRotateOp));

  // run dvpp
  APP_ERROR ret = AclAdapter::GetInstance().DvppRotate(input, output, degrees_, resample_, expand_, center_, fill);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppRotate: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppRotateOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  int32_t outputH = -1;
  int32_t outputW = -1;
  // if expand_, then we cannot know the shape. We need the input image to find the output shape --> set it to
  // <-1,-1[,3]>
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "DvppRotate: inputs cannot be empty.");
  if (inputs[0].Rank() < kMinImageRank) {
    std::string err_msg =
      "DvppRotate: input tensor should have at least 2 dimensions, but got: " + std::to_string(inputs[0].Rank());
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (!expand_) {
    outputH = static_cast<int32_t>(inputs[0][0]);
    outputW = static_cast<int32_t>(inputs[0][1]);
  }
  TensorShape out = TensorShape{outputH, outputW};
  if (inputs[0].Rank() == kMinImageRank) {
    (void)outputs.emplace_back(out);
  }
  if (inputs[0].Rank() == kDefaultImageRank) {
    (void)outputs.emplace_back(out.AppendDim(inputs[0][kChannelIndexHWC]));
  }
  if (inputs[0].Rank() > kDefaultImageRank) {
    auto out_shape = ConstructShape(inputs[0]);
    (void)outputs.emplace_back(out_shape);
  }
  return Status::OK();
}

Status DvppRotateOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  return Status::OK();
}

TensorShape DvppRotateOp::ConstructShape(const TensorShape &in_shape) {
  auto in_shape_vec = in_shape.AsVector();
  const int h_index = -3;
  const int w_index = -2;
  int32_t outputH = -1;
  int32_t outputW = -1;
  if (!expand_) {
    outputH = static_cast<int32_t>(in_shape[h_index]);
    outputW = static_cast<int32_t>(in_shape[w_index]);
  }
  in_shape_vec[in_shape_vec.size() + h_index] = outputH;
  in_shape_vec[in_shape_vec.size() + w_index] = outputW;
  TensorShape out = TensorShape(in_shape_vec);
  return out;
}
}  // namespace dataset
}  // namespace mindspore
