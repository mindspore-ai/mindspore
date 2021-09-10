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

#include "minddata/dataset/audio/ir/kernels/deemph_biquad_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/deemph_biquad_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// DeemphBiquadOperation
DeemphBiquadOperation::DeemphBiquadOperation(int32_t sample_rate) : sample_rate_(sample_rate) {}

Status DeemphBiquadOperation::ValidateParams() {
  if ((sample_rate_ != 44100 && sample_rate_ != 48000)) {
    std::string err_msg =
      "DeemphBiquad: sample_rate should be 44100 (hz) or 48000 (hz), but got: " + std::to_string(sample_rate_);
    MS_LOG(ERROR) << err_msg;
    return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> DeemphBiquadOperation::Build() {
  std::shared_ptr<DeemphBiquadOp> tensor_op = std::make_shared<DeemphBiquadOp>(sample_rate_);
  return tensor_op;
}

Status DeemphBiquadOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
