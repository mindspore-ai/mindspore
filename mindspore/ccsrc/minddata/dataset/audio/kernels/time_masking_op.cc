/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/audio/kernels/time_masking_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// constructor
TimeMaskingOp::TimeMaskingOp(bool iid_masks, int32_t time_mask_param, int32_t mask_start, float mask_value)
    : iid_masks_(iid_masks), time_mask_param_(time_mask_param), mask_start_(mask_start), mask_value_(mask_value) {
  rnd_.seed(GetSeed());
}

// main function
Status TimeMaskingOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // input <..., freq, time>
  RETURN_IF_NOT_OK(ValidateLowRank("TimeMasking", input, kDefaultAudioDim, "<..., freq, time>"));
  const int32_t kTimeIndex = -1;
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape()[kTimeIndex] >= time_mask_param_,
                               "TimeMasking: invalid parameter, 'time_mask_param' should be less than or equal to "
                               "the length of time dimension, but got: 'frequency_mask_param' " +
                                 std::to_string(time_mask_param_) + " and length " +
                                 std::to_string(input->shape()[kTimeIndex]));

  std::shared_ptr<Tensor> input_tensor;
  // typecast
  RETURN_IF_NOT_OK(ValidateTensorNumeric("TimeMasking", input));
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
  } else {
    input_tensor = input;
  }

  // iid_masks - whether to apply different masks to each example/channel.
  if (iid_masks_ == false) {
    return MaskAlongAxis(input_tensor, output, time_mask_param_, mask_start_, mask_value_, 2);
  } else {
    return RandomMaskAlongAxis(input_tensor, output, time_mask_param_, mask_value_, 2, rnd_);
  }
}

Status TimeMaskingOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("TimeMasking", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
