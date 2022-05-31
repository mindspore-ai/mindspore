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
#include "minddata/dataset/audio/kernels/vad_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status VadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("Vad", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Vad", input));
  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return Vad<float>(input_tensor, output, sample_rate_, trigger_level_, trigger_time_, search_time_, allowed_gap_,
                      pre_trigger_time_, boot_time_, noise_up_time_, noise_down_time_, noise_reduction_amount_,
                      measure_freq_, measure_duration_, measure_smooth_time_, hp_filter_freq_, lp_filter_freq_,
                      hp_lifter_freq_, lp_lifter_freq_);
  } else {
    input_tensor = input;
    return Vad<double>(input_tensor, output, sample_rate_, trigger_level_, trigger_time_, search_time_, allowed_gap_,
                       pre_trigger_time_, boot_time_, noise_up_time_, noise_down_time_, noise_reduction_amount_,
                       measure_freq_, measure_duration_, measure_smooth_time_, hp_filter_freq_, lp_filter_freq_,
                       hp_lifter_freq_, lp_lifter_freq_);
  }

  return Status::OK();
}

Status VadOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("Vad", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
