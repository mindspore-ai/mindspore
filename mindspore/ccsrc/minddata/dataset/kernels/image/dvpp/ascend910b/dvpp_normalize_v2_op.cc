/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_normalize_v2_op.h"

#include <utility>

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
constexpr int64_t h_lb = 4;     // height lower bound
constexpr int64_t h_ub = 8192;  // height upper bound
constexpr int64_t w_lb = 6;     // width lower bound
constexpr int64_t w_ub = 4096;  // width upper bound

DvppNormalizeV2Op::DvppNormalizeV2Op(std::vector<float> mean, std::vector<float> std, bool is_hwc)
    : mean_(std::move(mean)), std_(std::move(std)), is_hwc_(is_hwc) {}

Status DvppNormalizeV2Op::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                                  std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);

  // the input should be 1HWC
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetShape().Rank() == kNHWCImageRank,
                               "DvppNormalize: The input data's dims is not 4.");  // NHWC

  if (!is_hwc_) {
    CHECK_FAIL_RETURN_UNEXPECTED(
      input->GetShape().AsVector()[1] == kDefaultImageChannel || input->GetShape().AsVector()[1] == 1,
      "DvppNormalize: The input data's channel is not 3 or 1.");  // C == 3 or 1

    // the channel should be equal to the size of mean
    CHECK_FAIL_RETURN_UNEXPECTED(mean_.size() == std_.size() && std_.size() == input->GetShape().AsVector()[1],
                                 "DvppNormalize: The channel is not equal to the size of mean or std.");

    // Dvpp Limit
    int64_t input_h = input->GetShape()[kHeightIndexNCHW];
    int64_t input_w = input->GetShape()[kWidthIndexNCHW];
    RETURN_IF_NOT_OK(CheckDvppLimit(input_h, input_w, h_lb, w_lb, h_ub, w_ub, kDvppNormalizeOp));
  } else {
    CHECK_FAIL_RETURN_UNEXPECTED(input->GetShape().AsVector()[kChannelIndexNHWC] == kDefaultImageChannel ||
                                   input->GetShape().AsVector()[kChannelIndexNHWC] == 1,
                                 "DvppNormalize: The input data's channel is not 3 or 1.");  // C == 3 or 1

    // the channel should be equal to the size of mean
    CHECK_FAIL_RETURN_UNEXPECTED(
      mean_.size() == std_.size() && std_.size() == input->GetShape().AsVector()[kChannelIndexNHWC],
      "DvppNormalize: The channel is not equal to the size of mean or std.");

    // Dvpp Limit
    int64_t input_h = input->GetShape()[kHeightIndexNHWC];
    int64_t input_w = input->GetShape()[kWidthIndexNHWC];
    RETURN_IF_NOT_OK(CheckDvppLimit(input_h, input_w, h_lb, w_lb, h_ub, w_ub, kDvppNormalizeOp));
  }

  CHECK_FAIL_RETURN_UNEXPECTED(input->GetShape().AsVector()[0] == 1,
                               "DvppNormalize: The input data is not 1HWC.");  // N == 1

  // the type is uint8 / float
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetType() == DataType::DE_UINT8 || input->GetType() == DataType::DE_FLOAT32,
                               "DvppNormalize: The input data is not uint8 or float32.");

  APP_ERROR ret = AclAdapter::GetInstance().DvppNormalize(input, output, mean_, std_, is_hwc_);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppNormalize: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }

  return Status::OK();
}

void DvppNormalizeV2Op::Print(std::ostream &out) const {
  out << "DvppNormalizeOp, mean: ";
  for (const auto &m : mean_) {
    out << m << ", ";
  }
  out << "}" << std::endl << "std: ";
  for (const auto &s : std_) {
    out << s << ", ";
  }
  out << "}" << std::endl << "is_hwc: " << is_hwc_;
  out << "}" << std::endl;
}
}  // namespace dataset
}  // namespace mindspore
