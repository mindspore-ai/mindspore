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

#include "minddata/dataset/kernels/image/rotate_op.h"

#include "minddata/dataset/kernels/data/data_utils.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif

namespace mindspore {
namespace dataset {
const std::vector<float> RotateOp::kDefCenter = {};
const InterpolationMode RotateOp::kDefInterpolation = InterpolationMode::kNearestNeighbour;
const bool RotateOp::kDefExpand = false;
const uint8_t RotateOp::kDefFillR = 0;
const uint8_t RotateOp::kDefFillG = 0;
const uint8_t RotateOp::kDefFillB = 0;

RotateOp::RotateOp(int angle_id)
    : angle_id_(angle_id),
      degrees_(0),
      center_({}),
      interpolation_(InterpolationMode::kLinear),
      expand_(false),
      fill_r_(0),
      fill_g_(0),
      fill_b_(0) {}

RotateOp::RotateOp(float degrees, InterpolationMode resample, bool expand, std::vector<float> center, uint8_t fill_r,
                   uint8_t fill_g, uint8_t fill_b)
    : angle_id_(0),
      degrees_(degrees),
      center_(center),
      interpolation_(resample),
      expand_(expand),
      fill_r_(fill_r),
      fill_g_(fill_g),
      fill_b_(fill_b) {}

Status RotateOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImage(input, "Rotate", {1, 2, 3, 4, 5, 6, 10, 11, 12}));
  if (input->Rank() <= kDefaultImageRank) {
    // [H, W] or [H, W, C]
#ifndef ENABLE_ANDROID
    RETURN_IF_NOT_OK(Rotate(input, output, center_, degrees_, interpolation_, expand_, fill_r_, fill_g_, fill_b_));
#else
    RETURN_IF_NOT_OK(Rotate(input, output, angle_id_));
#endif
  } else {
    // reshape [..., H, W, C] to [N, H, W, C]
    auto original_shape = input->shape();
    dsize_t num_batch = input->Size() / (original_shape[-3] * original_shape[-2] * original_shape[-1]);
    TensorShape new_shape({num_batch, original_shape[-3], original_shape[-2], original_shape[-1]});
    RETURN_IF_NOT_OK(input->Reshape(new_shape));

    // split [N, H, W, C] to N [H, W, C], and Rotate N [H, W, C]
    std::vector<std::shared_ptr<Tensor>> input_vector_hwc, output_vector_hwc;
    RETURN_IF_NOT_OK(BatchTensorToTensorVector(input, &input_vector_hwc));
    for (auto input_hwc : input_vector_hwc) {
      std::shared_ptr<Tensor> output_img;
#ifndef ENABLE_ANDROID
      RETURN_IF_NOT_OK(
        Rotate(input_hwc, &output_img, center_, degrees_, interpolation_, expand_, fill_r_, fill_g_, fill_b_));
#else
      RETURN_IF_NOT_OK(Rotate(input_hwc, &output_img, angle_id_));
#endif
      output_vector_hwc.push_back(output_img);
    }
    // integrate N [H, W, C] to [N, H, W, C], and reshape [..., H, W, C]
    RETURN_IF_NOT_OK(TensorVectorToBatchTensor(output_vector_hwc, output));
    // reshape output before return, only height and width are changed
    auto output_shape_new = ConstructShape(original_shape);
    RETURN_IF_NOT_OK((*output)->Reshape(output_shape_new));
  }
  return Status::OK();
}

Status RotateOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
#ifndef ENABLE_ANDROID
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  int32_t outputH = -1, outputW = -1;
  // if expand_, then we cannot know the shape. We need the input image to find the output shape --> set it to
  // <-1,-1[,3]>
  if (!expand_) {
    outputH = inputs[0][0];
    outputW = inputs[0][1];
  }
  TensorShape out = TensorShape{outputH, outputW};
  if (inputs[0].Rank() < kMinImageRank) {
    std::string err_msg =
      "Rotate: input tensor should have at least 2 dimensions, but got: " + std::to_string(inputs[0].Rank());
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (inputs[0].Rank() == kMinImageRank) {
    (void)outputs.emplace_back(out);
  }
  if (inputs[0].Rank() == kDefaultImageRank) {
    outputs.emplace_back(out.AppendDim(inputs[0][kChannelIndexHWC]));
  }
  if (inputs[0].Rank() > kDefaultImageRank) {
    auto out_shape = ConstructShape(inputs[0]);
    (void)outputs.emplace_back(out_shape);
  }
  return Status::OK();
#else
  if (inputs.size() != NumInput()) {
    return Status(StatusCode::kMDUnexpectedError,
                  "The size of the input argument vector: " + std::to_string(inputs.size()) +
                    ", does not match the number of inputs: " + std::to_string(NumInput()));
  }
  outputs = inputs;
  return Status::OK();
#endif
}

TensorShape RotateOp::ConstructShape(const TensorShape &in_shape) {
  auto in_shape_vec = in_shape.AsVector();
  const int h_index = -3, w_index = -2;
  int32_t outputH = -1, outputW = -1;
  if (!expand_) {
    outputH = in_shape[h_index];
    outputW = in_shape[w_index];
  }
  in_shape_vec[in_shape_vec.size() + h_index] = outputH;
  in_shape_vec[in_shape_vec.size() + w_index] = outputW;
  TensorShape out = TensorShape(in_shape_vec);
  return out;
}
}  // namespace dataset
}  // namespace mindspore
