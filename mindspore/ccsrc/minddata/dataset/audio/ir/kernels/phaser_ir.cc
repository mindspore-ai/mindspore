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

#include "minddata/dataset/audio/ir/kernels/phaser_ir.h"

#include "minddata/dataset/audio/kernels/phaser_op.h"
#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace audio {
PhaserOperation::PhaserOperation(int32_t sample_rate, float gain_in, float gain_out, float delay_ms, float decay,
                                 float mod_speed, bool sinusoidal)
    : sample_rate_(sample_rate),
      gain_in_(gain_in),
      gain_out_(gain_out),
      delay_ms_(delay_ms),
      decay_(decay),
      mod_speed_(mod_speed),
      sinusoidal_(sinusoidal) {}

Status PhaserOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("Phaser", "gain_in", gain_in_, {0.0f, 1.0f}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Phaser", "gain_out", gain_out_, {0.0f, 1e9f}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Phaser", "delay_ms", delay_ms_, {0.0f, 5.0f}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Phaser", "decay", decay_, {0.0f, 0.99f}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Phaser", "mod_speed", mod_speed_, {0.1f, 2.0f}, false, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> PhaserOperation::Build() {
  std::shared_ptr<PhaserOp> tensor_op =
    std::make_shared<PhaserOp>(sample_rate_, gain_in_, gain_out_, delay_ms_, decay_, mod_speed_, sinusoidal_);
  return tensor_op;
}

Status PhaserOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["gain_in"] = gain_in_;
  args["gain_out"] = gain_out_;
  args["delay_ms"] = delay_ms_;
  args["decay"] = decay_;
  args["mod_speed"] = mod_speed_;
  args["sinusoidal"] = sinusoidal_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
