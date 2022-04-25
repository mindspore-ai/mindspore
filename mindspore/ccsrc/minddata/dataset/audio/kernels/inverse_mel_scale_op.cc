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
#include "minddata/dataset/audio/kernels/inverse_mel_scale_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status InverseMelScaleOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  // check and init
  IO_CHECK(input, output);
  // check input dimension, it should be greater than 0
  RETURN_IF_NOT_OK(ValidateLowRank("InverseMelScale", input, kDefaultAudioDim, "<..., freq, time>"));
  // check input type, it should be [int, float, double]
  RETURN_IF_NOT_OK(ValidateTensorNumeric("InverseMelScale", input));

  return InverseMelScale(input, output, n_stft_, n_mels_, sample_rate_, f_min_, f_max_, max_iter_, tolerance_loss_,
                         tolerance_change_, sgd_lr_, sgd_momentum_, norm_, mel_type_, rnd_);
}

Status InverseMelScaleOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  auto input_size = inputs[0].AsVector();
  input_size.pop_back();
  TensorShape out = TensorShape(input_size);
  outputs.emplace_back(out);
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kMDUnexpectedError, "InverseMelScale: invalid input shape.");
}

Status InverseMelScaleOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("InverseMelScale", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
