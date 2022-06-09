/**
 * Copyright 2022 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/audio/kernels/resample_op.h"

#include <cmath>

#include "minddata/dataset/audio/kernels/audio_utils.h"

namespace mindspore {
namespace dataset {
// main function call for resample
Status ResampleOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (orig_freq_ == new_freq_) {
    *output = input;
  } else {
    RETURN_IF_NOT_OK(
      Resample(input, output, orig_freq_, new_freq_, resample_method_, lowpass_filter_width_, rolloff_, beta_));
  }
  return Status::OK();
}

Status ResampleOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();

  auto input_size = inputs[0].AsVector();
  input_size.pop_back();
  int32_t waveform_length = inputs[0][-1];
  int32_t gcd = std::gcd(static_cast<int32_t>(orig_freq_), static_cast<int32_t>(new_freq_));
  CHECK_FAIL_RETURN_UNEXPECTED(gcd != 0, "Resample: gcd cannet be equal to 0.");
  int32_t orig_freq = static_cast<int32_t>(floor(orig_freq_ / gcd));
  int32_t new_freq = static_cast<int32_t>(floor(new_freq_ / gcd));
  int32_t target_length = static_cast<int32_t>(std::ceil(static_cast<float>(new_freq * waveform_length) / orig_freq));
  input_size.at(input_size.size() - 1) = target_length;
  TensorShape out = TensorShape(input_size);
  outputs.emplace_back(out);
  if (outputs.empty()) {
    return Status(StatusCode::kMDUnexpectedError, "Resample: invalid shape of input shape.");
  } else {
    return Status::OK();
  }
}

Status ResampleOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("Resample", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
