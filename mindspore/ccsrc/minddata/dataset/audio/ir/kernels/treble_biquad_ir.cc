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
#include "minddata/dataset/audio/ir/kernels/treble_biquad_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/treble_biquad_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
TrebleBiquadOperation::TrebleBiquadOperation(int32_t sample_rate, float gain, float central_freq, float Q)
    : sample_rate_(sample_rate), gain_(gain), central_freq_(central_freq), Q_(Q) {}

Status TrebleBiquadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("TrebleBiquad", "Q", Q_, {0, 1.0}, true, false));
  RETURN_IF_NOT_OK(ValidateScalarNotZero("TrebleBiquad", "sample_rate", sample_rate_));
  return Status::OK();
}

std::shared_ptr<TensorOp> TrebleBiquadOperation::Build() {
  std::shared_ptr<TrebleBiquadOp> tensor_op = std::make_shared<TrebleBiquadOp>(sample_rate_, gain_, central_freq_, Q_);
  return tensor_op;
}

Status TrebleBiquadOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["gain"] = gain_;
  args["central_freq"] = central_freq_;
  args["Q"] = Q_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
