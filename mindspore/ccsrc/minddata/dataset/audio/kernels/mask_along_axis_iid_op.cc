/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/audio/kernels/mask_along_axis_iid_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
const int32_t kFrequencyAxis = 1;
const int32_t kTimeAxis = 2;
const int32_t kTensorFreqiencyPos = -2;
const int32_t kTensorTimePos = -1;

Status MaskAlongAxisIIDOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  RETURN_IF_NOT_OK(ValidateLowRank("MaskAlongAxisIID", input, kDefaultAudioDim, "<..., freq, time>"));
  RETURN_IF_NOT_OK(ValidateTensorType("MaskAlongAxisIID", input->type().IsNumeric(), "[int, float, double]",
                                      input->type().ToString()));
  TensorShape input_shape = input->shape();

  if (axis_ == kFrequencyAxis) {
    CHECK_FAIL_RETURN_UNEXPECTED(
      input_shape[kTensorFreqiencyPos] >= mask_param_,
      "MaskAlongAxisIID: mask_param should be less than or equal to the length of frequency dimension.");
  } else if (axis_ == kTimeAxis) {
    CHECK_FAIL_RETURN_UNEXPECTED(
      input_shape[kTensorTimePos] >= mask_param_,
      "MaskAlongAxisIID: mask_param should be less than or equal to the length of time dimension.");
  } else {
    RETURN_STATUS_UNEXPECTED("MaskAlongAxisIID: only support Frequency and Time masking, axis should be 1 or 2.");
  }

  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
  } else {
    input_tensor = input;
  }
  return RandomMaskAlongAxis(input_tensor, output, mask_param_, mask_value_, axis_, rnd_);
}

Status MaskAlongAxisIIDOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("MaskAlongAxisIID", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));

  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
