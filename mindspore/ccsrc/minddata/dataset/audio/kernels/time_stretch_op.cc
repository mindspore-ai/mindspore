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
#include "minddata/dataset/audio/kernels/time_stretch_op.h"

#include <limits>

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const float TimeStretchOp::kHopLength = std::numeric_limits<float>::quiet_NaN();
const int TimeStretchOp::kNFreq = 201;
const float TimeStretchOp::kFixedRate = std::numeric_limits<float>::quiet_NaN();

Status TimeStretchOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  // check and init
  IO_CHECK(input, output);

  // check shape
  if (input->shape().Rank() < 3 || !input->IsComplex()) {
    std::string err_msg = "TimeStretch: input tensor is not in shape of <..., freq, num_frame, complex=2>.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::shared_ptr<Tensor> input_tensor;
  // std::shared_ptr<Tensor> phase_advance;
  float hop_length = std::isnan(hop_length_) ? (n_freq_ - 1) : hop_length_;
  // typecast
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() != DataType::DE_STRING,
                               "TimeStretch: input tensor type should be int, float or double, but got: string.");
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
  } else {
    input_tensor = input;
  }

  return TimeStretch(input_tensor, output, fixed_rate_, hop_length, n_freq_);
}

Status TimeStretchOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  for (auto s : inputs) {
    std::vector<dsize_t> s_vec = s.AsVector();
    s_vec.pop_back();
    s_vec.pop_back();
    s_vec.push_back(std::ceil(s[-2] / fixed_rate_));
    // push back complex
    s_vec.push_back(2);
    outputs.emplace_back(TensorShape(s_vec));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!outputs.empty(), "TimeStretch: invalid input shape.");
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
