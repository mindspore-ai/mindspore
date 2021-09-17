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

#include "minddata/dataset/audio/ir/kernels/bandpass_biquad_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/bandpass_biquad_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// BandpassBiquadOperation
BandpassBiquadOperation::BandpassBiquadOperation(int32_t sample_rate, float central_freq, float Q,
                                                 bool const_skirt_gain)
    : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q), const_skirt_gain_(const_skirt_gain) {}

Status BandpassBiquadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("BandpassBiquad", "Q", Q_, {0, 1.0}, true, false));
  RETURN_IF_NOT_OK(ValidateScalarNotZero("BandpassBiquad", "sample_rate", sample_rate_));
  return Status::OK();
}

std::shared_ptr<TensorOp> BandpassBiquadOperation::Build() {
  std::shared_ptr<BandpassBiquadOp> tensor_op =
    std::make_shared<BandpassBiquadOp>(sample_rate_, central_freq_, Q_, const_skirt_gain_);
  return tensor_op;
}

Status BandpassBiquadOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["central_freq"] = central_freq_;
  args["Q"] = Q_;
  args["const_skirt_gain"] = const_skirt_gain_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
