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
#include "minddata/dataset/audio/kernels/amplitude_to_db_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status AmplitudeToDBOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("AmplitudeToDB", input, kDefaultAudioDim, "<..., freq, time>"));

  std::shared_ptr<Tensor> input_tensor;

  float top_db = top_db_;
  float multiplier = stype_ == ScaleType::kPower ? 10.0 : 20.0;
  const float amin = 1e-10;
  float db_multiplier = std::log10(std::max(amin_, ref_value_));

  RETURN_IF_NOT_OK(ValidateTensorNumeric("AmplitudeToDB", input));
  // typecast
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return AmplitudeToDB<float>(input_tensor, output, multiplier, amin, db_multiplier, top_db);
  } else {
    input_tensor = input;
    return AmplitudeToDB<double>(input_tensor, output, multiplier, amin, db_multiplier, static_cast<double>(top_db));
  }
}
}  // namespace dataset
}  // namespace mindspore
