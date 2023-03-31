/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "utils/ms_utils.h"

#include "minddata/dataset/kernels/data/data_utils.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif

#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const int32_t CenterCropOp::kDefWidth = 0;

Status CenterCropOp::CenterCropImg(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  int32_t top = crop_het_ - input->shape()[0];  // number of pixels to pad (top and bottom)
  int32_t left = crop_wid_ - input->shape()[1];
  std::shared_ptr<Tensor> pad_image;

  constexpr int64_t pad_limit = 3;
  CHECK_FAIL_RETURN_UNEXPECTED((top < input->shape()[0] * pad_limit && left < input->shape()[1] * pad_limit),
                               "CenterCrop: CenterCropOp padding size is more than 3 times the original size, got pad"
                               " top: " +
                                 std::to_string(top) + "pad left: " + std::to_string(left) + ", and original size: " +
                                 std::to_string(input->shape()[0]) + ", " + std::to_string(input->shape()[1]));

  if (top > 0 && left > 0) {  // padding only
    return Pad(input, output, top / 2 + top % 2, top / 2, left / 2 + left % 2, left / 2, BorderType::kConstant);
  } else if (top > 0) {
    RETURN_IF_NOT_OK(Pad(input, &pad_image, top / 2 + top % 2, top / 2, 0, 0, BorderType::kConstant));
    return Crop(pad_image, output, (static_cast<int32_t>(pad_image->shape()[1]) - crop_wid_) / 2,
                (static_cast<int32_t>(pad_image->shape()[0]) - crop_het_) / 2, crop_wid_, crop_het_);
  } else if (left > 0) {
    RETURN_IF_NOT_OK(Pad(input, &pad_image, 0, 0, left / 2 + left % 2, left / 2, BorderType::kConstant));
    return Crop(pad_image, output, (static_cast<int32_t>(pad_image->shape()[1]) - crop_wid_) / 2,
                (static_cast<int32_t>(pad_image->shape()[0]) - crop_het_) / 2, crop_wid_, crop_het_);
  }
  return Crop(input, output, (input->shape()[1] - crop_wid_) / 2, (input->shape()[0] - crop_het_) / 2, crop_wid_,
              crop_het_);
}

Status CenterCropOp::ConstructShape(const TensorShape &in_shape, std::shared_ptr<TensorShape> *out_shape) {
  auto in_shape_vec = in_shape.AsVector();
  const int h_index = -3, w_index = -2;
  in_shape_vec[in_shape_vec.size() + h_index] = crop_het_;
  in_shape_vec[in_shape_vec.size() + w_index] = crop_wid_;

  *out_shape = std::make_shared<TensorShape>(in_shape_vec);

  return Status::OK();
}

Status CenterCropOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  std::string err_msg;
  std::string err_head = "CenterCrop: ";
  dsize_t rank = input->shape().Rank();

  err_msg += (rank < kMinImageRank)
               ? "input tensor should have at least 2 dimensions, but got: " + std::to_string(rank) + "\t"
               : "";
  err_msg += (crop_het_ <= 0 || crop_wid_ <= 0)
               ? "crop size needs to be positive integers, but got crop height:" + std::to_string(crop_het_) +
                   ", crop width: " + std::to_string(crop_wid_) + "\t"
               : "";
  CHECK_FAIL_RETURN_SYNTAX_ERROR(err_msg.length() == 0, err_head + err_msg);

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
    for (int i = 0; i < num_batch; i++) {
      std::shared_ptr<Tensor> center_crop;
      RETURN_IF_NOT_OK(CenterCropImg(input_vector_hwc[i], &center_crop));
      output_vector_hwc.push_back(center_crop);
    }

    // integrate N [H, W, C] to [N, H, W, C], and reshape [..., H, W, C]
    RETURN_IF_NOT_OK(TensorVectorToBatchTensor(output_vector_hwc, output));
    // reshape output before return, only height and width are changed
    std::shared_ptr<TensorShape> output_shape_new = nullptr;
    RETURN_IF_NOT_OK(ConstructShape(input_shape, &output_shape_new));
    RETURN_IF_NOT_OK((*output)->Reshape(*output_shape_new));
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
  if (inputs[0].Rank() == kMinImageRank) {
    (void)outputs.emplace_back(out);
  }
  if (inputs[0].Rank() == kDefaultImageRank) {
    (void)outputs.emplace_back(out.AppendDim(inputs[0][kChannelIndexHWC]));
  }
  if (inputs[0].Rank() > kDefaultImageRank) {
    std::shared_ptr<TensorShape> output_shape_new = nullptr;
    RETURN_IF_NOT_OK(ConstructShape(inputs[0], &output_shape_new));
    (void)outputs.emplace_back(*output_shape_new);
  }
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(
    StatusCode::kMDUnexpectedError,
    "CenterCrop: input tensor should have at least 2 dimensions, but got: " + std::to_string(inputs[0].Rank()));
}
}  // namespace dataset
}  // namespace mindspore
