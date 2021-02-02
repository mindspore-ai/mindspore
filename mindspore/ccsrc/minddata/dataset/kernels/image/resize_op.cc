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
#include "minddata/dataset/kernels/image/resize_op.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const int32_t ResizeOp::kDefWidth = 0;
const InterpolationMode ResizeOp::kDefInterpolation = InterpolationMode::kLinear;

Status ResizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape().Size() >= 2, "Resize: image shape is not <H,W,C> or <H,W>.");
  int32_t output_h, output_w = 0;
  int32_t input_h = static_cast<int>(input->shape()[0]);
  int32_t input_w = static_cast<int>(input->shape()[1]);
  if (size2_ == 0) {
    if (input_h < input_w) {
      CHECK_FAIL_RETURN_UNEXPECTED(input_h != 0, "Resize: the input height is 0.");
      output_h = size1_;
      output_w = static_cast<int>(std::lround(static_cast<float>(input_w) / input_h * output_h));
    } else {
      CHECK_FAIL_RETURN_UNEXPECTED(input_w != 0, "Resize: the input width is 0.");
      output_w = size1_;
      output_h = static_cast<int>(std::lround(static_cast<float>(input_h) / input_w * output_w));
    }
  } else {
    output_h = size1_;
    output_w = size2_;
  }
  return Resize(input, output, output_h, output_w, 0, 0, interpolation_);
}

Status ResizeOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  int32_t outputH = -1, outputW = -1;
  // if size2_ == 0, then we cannot know the shape. We need the input image to find the output shape --> set it to
  // <-1,-1[,3]>
  if (size2_ != 0) {
    outputH = size1_;
    outputW = size2_;
  }
  TensorShape out = TensorShape{outputH, outputW};
  if (inputs[0].Rank() == 2) outputs.emplace_back(out);
  if (inputs[0].Rank() == 3) outputs.emplace_back(out.AppendDim(inputs[0][2]));
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kMDUnexpectedError, "Resize: invalid input wrong shape.");
}
}  // namespace dataset
}  // namespace mindspore
