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
#include "minddata/dataset/kernels/image/decode_video_op.h"

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/video_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
DecodeVideoOp::DecodeVideoOp() {}

Status DecodeVideoOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  // check the input tensor shape
  if (input[0]->Rank() != 1) {
    RETURN_STATUS_UNEXPECTED("DecodeVideo: invalid input shape, only support 1D input, got rank: " +
                             std::to_string(input[0]->Rank()));
  }
  return DecodeVideo(input, output);
}

Status DecodeVideoOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  // kDefaultImageRank is 3
  TensorShape visual_shape({-1, -1, -1, kDefaultImageRank});
  TensorShape audio_shape({-1, -1});
  if (inputs[0].Rank() == 1) {
    outputs.emplace_back(visual_shape);
    outputs.emplace_back(audio_shape);
  } else {
    RETURN_STATUS_UNEXPECTED("DecodeVideo: invalid input shape, expected 1D input, but got input dimension: " +
                             std::to_string(inputs[0].Rank()));
  }
  return Status::OK();
}

Status DecodeVideoOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  outputs[0] = DataType(DataType::DE_UINT8);
  outputs[1] = DataType(DataType::DE_UNKNOWN);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
