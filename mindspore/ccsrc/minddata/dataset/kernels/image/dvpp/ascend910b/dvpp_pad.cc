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
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_pad.h"

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
DvppPadOp::DvppPadOp(int32_t pad_top, int32_t pad_bottom, int32_t pad_left, int32_t pad_right, BorderType padding_mode,
                     uint8_t fill_r, uint8_t fill_g, uint8_t fill_b)
    : pad_top_(pad_top),
      pad_bottom_(pad_bottom),
      pad_left_(pad_left),
      pad_right_(pad_right),
      boarder_type_(padding_mode),
      fill_r_(fill_r),
      fill_g_(fill_g),
      fill_b_(fill_b) {}

Status DvppPadOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                          std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);

  // the input should be NHWC, N is 1.
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->GetShape().Rank() == kNHWCImageRank,
    "DvppPad: the input tensor is not HW, HWC or 1HWC, but got: " + std::to_string(input->GetShape().Rank()));

  std::vector<int64_t> padding = {pad_left_, pad_top_, pad_right_, pad_bottom_};
  std::vector<float> fill = {static_cast<float>(fill_r_), static_cast<float>(fill_g_), static_cast<float>(fill_b_)};
  uint32_t padding_mode = GetDVPPPaddingMode(boarder_type_);

  if (padding_mode == kInvalidPaddingMode) {
    auto error =
      "DvppPad: Invalid padding mode, only support Border.CONSTANT, Border.EDGE, Border.REFLECT and Border.SYMMETRIC";
    RETURN_STATUS_UNEXPECTED(error);
  }

  // Dvpp Limit
  int64_t input_h = input->GetShape()[kHeightIndexNHWC];
  int64_t input_w = input->GetShape()[kWidthIndexNHWC];
  int64_t output_h = input_h + pad_top_ + pad_bottom_;
  int64_t output_w = input_w + pad_left_ + pad_right_;

  constexpr int32_t h_lb = 4;       // height lower bound
  constexpr int32_t h_ub = 32768;   // height upper bound
  constexpr int32_t w_lb = 6;       // width lower bound
  constexpr int32_t w_ub = 32768;   // width upper bound
  constexpr int32_t pad_ub = 2048;  // max padding number

  if ((input_h < h_lb || input_h > h_ub) || (input_w < w_lb || input_w > w_ub)) {
    auto error = "DvppPad: due to hardware limit, the input shape should be from [4, 6] to [32768, 32768], but got [" +
                 std::to_string(input_h) + ", " + std::to_string(input_w) + "].";
    RETURN_STATUS_UNEXPECTED(error);
  }
  if ((output_h < h_lb || output_h > h_ub) || (output_w < w_lb || output_w > w_ub)) {
    auto error = "DvppPad: due to hardware limit, the output shape should be from [4, 6] to [32768, 32768], but got [" +
                 std::to_string(output_h) + ", " + std::to_string(output_w) + "].";
    RETURN_STATUS_UNEXPECTED(error);
  }
  for (const int64_t &p : padding) {
    if (p > pad_ub) {
      auto error =
        "DvppPad: due to hardware limit, the padding pixel number should be less than or equal to 2048, but got" +
        std::to_string(p);
    }
  }

  // run dvpp
  APP_ERROR ret = AclAdapter::GetInstance().DvppPad(input, output, padding, padding_mode, fill);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppPad: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppPadOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  outputs.clear();
  int32_t height = inputs[0][kHeightIndex];
  int32_t width = inputs[0][kWidthIndex];
  TensorShape out = TensorShape{height + pad_top_ + pad_bottom_, width + pad_left_ + pad_right_};

  if (inputs[0].Rank() == kMinImageRank) {
    (void)outputs.emplace_back(out);
  }
  if (inputs[0].Rank() == kDefaultImageRank) {
    (void)outputs.emplace_back(out.AppendDim(inputs[0][kChannelIndexHWC]));
  }
  return Status::OK();
}

Status DvppPadOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
