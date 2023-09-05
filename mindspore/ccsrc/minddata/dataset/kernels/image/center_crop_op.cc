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
#include "minddata/dataset/kernels/image/center_crop_op.h"

#include <string>

#include "minddata/dataset/kernels/data/data_utils.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/status.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
const int32_t CenterCropOp::kDefWidth = 0;

Status CenterCropOp::CenterCropImg(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) const {
  RETURN_UNEXPECTED_IF_NULL(output);
  int32_t top = crop_het_ - input->shape()[0];  // number of pixels to pad (top and bottom)
  int32_t left = crop_wid_ - input->shape()[1];
  std::shared_ptr<Tensor> pad_image;

  const int64_t kMaxPadScale = 3;
  CHECK_FAIL_RETURN_UNEXPECTED(
    top < input->shape()[0] * kMaxPadScale && left < input->shape()[1] * kMaxPadScale,
    "CenterCrop: Padding size cannot be more than 3 times of the original image size, got top padding: " +
      std::to_string(top) + ", left padding: " + std::to_string(left) + ", while the original image size: " +
      std::to_string(input->shape()[0]) + ", " + std::to_string(input->shape()[1]));

  const int32_t kDivisorOfHalf = 2;
  if (top > 0 && left > 0) {
    return Pad(input, output, (top + 1) / kDivisorOfHalf, top / kDivisorOfHalf, (left + 1) / kDivisorOfHalf,
               left / kDivisorOfHalf, BorderType::kConstant);
  } else if (top > 0) {
    RETURN_IF_NOT_OK(
      Pad(input, &pad_image, (top + 1) / kDivisorOfHalf, top / kDivisorOfHalf, 0, 0, BorderType::kConstant));
    return Crop(pad_image, output, (static_cast<int32_t>(pad_image->shape()[1]) - crop_wid_) / kDivisorOfHalf,
                (static_cast<int32_t>(pad_image->shape()[0]) - crop_het_) / kDivisorOfHalf, crop_wid_, crop_het_);
  } else if (left > 0) {
    RETURN_IF_NOT_OK(
      Pad(input, &pad_image, 0, 0, (left + 1) / kDivisorOfHalf, left / kDivisorOfHalf, BorderType::kConstant));
    return Crop(pad_image, output, (static_cast<int32_t>(pad_image->shape()[1]) - crop_wid_) / kDivisorOfHalf,
                (static_cast<int32_t>(pad_image->shape()[0]) - crop_het_) / kDivisorOfHalf, crop_wid_, crop_het_);
  } else {
    return Crop(input, output, (static_cast<int32_t>(input->shape()[1]) - crop_wid_) / kDivisorOfHalf,
                (static_cast<int32_t>(input->shape()[0]) - crop_het_) / kDivisorOfHalf, crop_wid_, crop_het_);
  }
}

Status CenterCropOp::ConstructShape(const TensorShape &in_shape, std::shared_ptr<TensorShape> *out_shape) const {
  RETURN_UNEXPECTED_IF_NULL(out_shape);
  auto in_shape_vec = in_shape.AsVector();
  const int h_index = -3;
  const int w_index = -2;
  in_shape_vec[in_shape_vec.size() + h_index] = crop_het_;
  in_shape_vec[in_shape_vec.size() + w_index] = crop_wid_;

  *out_shape = std::make_shared<TensorShape>(in_shape_vec);

  return Status::OK();
}

Status CenterCropOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  dsize_t rank = input->shape().Rank();
  CHECK_FAIL_RETURN_UNEXPECTED(
    rank >= kMinImageRank,
    "CenterCrop: Input tensor should have at least 2 dimensions, but got: " + std::to_string(rank));
  CHECK_FAIL_RETURN_UNEXPECTED(crop_het_ > 0 && crop_wid_ > 0,
                               "CenterCrop: Crop size should be positive, but got crop height: " +
                                 std::to_string(crop_het_) + ", crop width: " + std::to_string(crop_wid_));

  if (rank <= kDefaultImageRank) {  // images
    RETURN_IF_NOT_OK(CenterCropImg(input, output));
  } else {  // deal with videos
    // reshape input to nhwc
    auto input_shape = input->shape();
    dsize_t num_batch = input->Size() / (input_shape[-3] * input_shape[-2] * input_shape[-1]);
    TensorShape new_shape({num_batch, input_shape[-3], input_shape[-2], input_shape[-1]});
    RETURN_IF_NOT_OK(input->Reshape(new_shape));

    // split [N, H, W, C] to N [H, W, C], and center crop N [H, W, C]
    std::vector<std::shared_ptr<Tensor>> input_vector_hwc, output_vector_hwc;
    RETURN_IF_NOT_OK(BatchTensorToTensorVector(input, &input_vector_hwc));
    for (auto &image : input_vector_hwc) {
      std::shared_ptr<Tensor> center_crop;
      RETURN_IF_NOT_OK(CenterCropImg(image, &center_crop));
      output_vector_hwc.push_back(center_crop);
    }

    // integrate N [H, W, C] to [N, H, W, C], and reshape [..., H, W, C]
    RETURN_IF_NOT_OK(TensorVectorToBatchTensor(output_vector_hwc, output));
    // reshape output before return, only height and width are changed
    std::shared_ptr<TensorShape> output_shape;
    RETURN_IF_NOT_OK(ConstructShape(input_shape, &output_shape));
    RETURN_IF_NOT_OK((*output)->Reshape(*output_shape));
  }

  return Status::OK();
}

void CenterCropOp::Print(std::ostream &out) const {
  out << "CenterCropOp: "
      << "cropWidth: " << crop_wid_ << "cropHeight: " << crop_het_ << "\n";
}

Status CenterCropOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out = TensorShape{crop_het_, crop_wid_};
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "CenterCrop: inputs cannot be empty.");
  if (inputs[0].Rank() == kMinImageRank) {
    (void)outputs.emplace_back(out);
  } else if (inputs[0].Rank() == kDefaultImageRank) {
    (void)outputs.emplace_back(out.AppendDim(inputs[0][kChannelIndexHWC]));
  } else if (inputs[0].Rank() > kDefaultImageRank) {
    std::shared_ptr<TensorShape> output_shape;
    RETURN_IF_NOT_OK(ConstructShape(inputs[0], &output_shape));
    (void)outputs.emplace_back(*output_shape);
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    !outputs.empty(),
    "CenterCrop: input tensor should have at least 2 dimensions, but got: " + std::to_string(inputs[0].Rank()));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
