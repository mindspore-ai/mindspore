/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <vector>

#include "minddata/dataset/kernels/data/data_utils.h"
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
  RETURN_IF_NOT_OK(ValidateImage(input, "Resize", {1, 2, 3, 4, 5, 6, 10, 11, 12}));
  std::vector<dsize_t> size;
  RETURN_IF_NOT_OK(ImageSize(input, &size));
  int32_t input_h = size[kHeightIndex];
  int32_t input_w = size[kWidthIndex];
  int32_t output_h;
  int32_t output_w;
  if (size2_ == 0) {
    if (input_h < input_w) {
      CHECK_FAIL_RETURN_UNEXPECTED(input_h != 0, "Resize: the input height cannot be 0.");
      output_h = size1_;
      output_w = static_cast<int>(std::floor(static_cast<float>(input_w) / input_h * output_h));
    } else {
      CHECK_FAIL_RETURN_UNEXPECTED(input_w != 0, "Resize: the input width cannot be 0.");
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
  // [H, W] or [H, W, C]
  if (input->Rank() <= kDefaultImageRank) {
    RETURN_IF_NOT_OK(Resize(input, output, output_h, output_w, 0, 0, interpolation_));
  } else {
    // reshape [..., H, W, C] to [N, H, W, C]
    TensorShape original_shape = input->shape();
    dsize_t num_batch = input->Size() / (original_shape[-3] * original_shape[-2] * original_shape[-1]);
    TensorShape new_shape({num_batch, original_shape[-3], original_shape[-2], original_shape[-1]});
    RETURN_IF_NOT_OK(input->Reshape(new_shape));

    // split [N, H, W, C] to N [H, W, C], and Resize N [H, W, C]
    std::vector<std::shared_ptr<Tensor>> input_vector_hwc, output_vector_hwc;
    RETURN_IF_NOT_OK(BatchTensorToTensorVector(input, &input_vector_hwc));
    for (auto input_hwc : input_vector_hwc) {
      std::shared_ptr<Tensor> output_img;
      RETURN_IF_NOT_OK(Resize(input_hwc, &output_img, output_h, output_w, 0, 0, interpolation_));
      output_vector_hwc.push_back(output_img);
    }
    // integrate N [H, W, C] to [N, H, W, C], and reshape [..., H, W, C]
    RETURN_IF_NOT_OK(TensorVectorToBatchTensor(output_vector_hwc, &(*output)));
    auto output_shape = ComputeOutputShape(original_shape, output_h, output_w);
    RETURN_IF_NOT_OK((*output)->Reshape(output_shape));
  }
  return Status::OK();
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
  if (inputs[0].Rank() < kMinImageRank) {
    std::string err_msg =
      "Resize: input tensor should have at least 2 dimensions, but got: " + std::to_string(inputs[0].Rank());
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

TensorShape ResizeOp::ComputeOutputShape(const TensorShape &input, int32_t output_h, int32_t output_w) {
  const int kHeightIndex = -3, kWidthIndex = -2;
  auto out_shape_vec = input.AsVector();
  auto size = out_shape_vec.size();
  out_shape_vec[size + kHeightIndex] = output_h;
  out_shape_vec[size + kWidthIndex] = output_w;
  TensorShape out = TensorShape(out_shape_vec);
  return out;
}
}  // namespace dataset
}  // namespace mindspore
