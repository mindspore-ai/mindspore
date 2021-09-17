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

#include "minddata/dataset/audio/ir/kernels/equalizer_biquad_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/audio/kernels/equalizer_biquad_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
EqualizerBiquadOperation::EqualizerBiquadOperation(int32_t sample_rate, float center_freq, float gain, float Q)
    : sample_rate_(sample_rate), center_freq_(center_freq), gain_(gain), Q_(Q) {}

Status EqualizerBiquadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalarNotZero("EqualizerBiquad", "sample_rate", sample_rate_));
  RETURN_IF_NOT_OK(ValidateScalar("EqualizerBiquad", "Q", Q_, {0, 1.0}, true, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> EqualizerBiquadOperation::Build() {
  std::shared_ptr<EqualizerBiquadOp> tensor_op =
    std::make_shared<EqualizerBiquadOp>(sample_rate_, center_freq_, gain_, Q_);
  return tensor_op;
}

Status EqualizerBiquadOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["center_freq"] = center_freq_;
  args["gain"] = gain_;
  args["Q"] = Q_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
