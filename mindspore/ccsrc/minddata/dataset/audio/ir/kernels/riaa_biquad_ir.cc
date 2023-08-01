/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/audio/ir/kernels/riaa_biquad_ir.h"

#include "minddata/dataset/audio/kernels/riaa_biquad_op.h"
#include "minddata/dataset/audio/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace audio {
RiaaBiquadOperation::RiaaBiquadOperation(int32_t sample_rate) : sample_rate_(sample_rate) {}

Status RiaaBiquadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalarValue("RiaaBiquad", "sample_rate", sample_rate_, {44100, 48000, 88200, 96000}));
  return Status::OK();
}

std::shared_ptr<TensorOp> RiaaBiquadOperation::Build() {
  std::shared_ptr<RiaaBiquadOp> tensor_op = std::make_shared<RiaaBiquadOp>(sample_rate_);
  return tensor_op;
}

Status RiaaBiquadOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
