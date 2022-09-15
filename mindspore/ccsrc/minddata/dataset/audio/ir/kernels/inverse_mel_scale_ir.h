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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_INVERSE_MEL_SCALE_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_INVERSE_MEL_SCALE_IR_H_

#include <map>
#include <memory>
#include <string>

#include "include/api/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
namespace audio {
constexpr char kInverseMelScaleOperation[] = "InverseMelScale";

class InverseMelScaleOperation : public TensorOperation {
 public:
  InverseMelScaleOperation(int32_t n_stft, int32_t n_mels, int32_t sample_rate, float f_min, float f_max,
                           int32_t max_iter, float tolerance_loss, float tolerance_change,
                           const std::map<std::string, float> &sgdargs, NormType norm, MelType mel_type);

  ~InverseMelScaleOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t n_stft_;
  int32_t n_mels_;
  int32_t sample_rate_;
  float f_min_;
  float f_max_;
  int32_t max_iter_;
  float tolerance_loss_;
  float tolerance_change_;
  std::map<std::string, float> sgdargs_;
  float sgd_lr_;
  float sgd_momentum_;
  NormType norm_;
  MelType mel_type_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_INVERSE_MEL_SCALE_IR_H_
