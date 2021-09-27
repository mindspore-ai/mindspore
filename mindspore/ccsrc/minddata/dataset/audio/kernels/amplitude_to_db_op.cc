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
  if (input->shape().Rank() < 2) {
    std::string err_msg = "AmplitudeToDB: input tensor is not in shape of <..., freq, time>.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::shared_ptr<Tensor> input_tensor;

  float top_db = top_db_;
  float multiplier = stype_ == ScaleType::kPower ? 10.0 : 20.0;
  const float amin = 1e-10;
  float db_multiplier = std::log10(std::max(amin_, ref_value_));

  // typecast
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() != DataType::DE_STRING,
                               "AmplitudeToDB: input tensor type should be float, but got: string.");
  if (input->type() != DataType::DE_FLOAT64) {
    CHECK_FAIL_RETURN_UNEXPECTED(
      TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)),
      "AmplitudeToDB: input tensor type should be float, but got: " + input->type().ToString());
    return AmplitudeToDB<float>(input_tensor, output, multiplier, amin, db_multiplier, top_db);
  } else {
    input_tensor = input;
    return AmplitudeToDB<double>(input_tensor, output, multiplier, amin, db_multiplier, top_db);
  }
}
}  // namespace dataset
}  // namespace mindspore
