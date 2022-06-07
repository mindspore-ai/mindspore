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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_RESAMPLE_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_RESAMPLE_IR_H_

#include <memory>
#include <string>

#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
namespace audio {
constexpr char kResampleOperation[] = "Resample";

class ResampleOperation : public TensorOperation {
 public:
  ResampleOperation(float orig_freq, float new_freq, ResampleMethod resample_method, int32_t lowpass_filter_width,
                    float rolloff, float beta);

  ~ResampleOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  float orig_freq_;
  float new_freq_;
  ResampleMethod resample_method_;
  int32_t lowpass_filter_width_;
  float rolloff_;
  float beta_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_RESAMPLE_IR_H_
