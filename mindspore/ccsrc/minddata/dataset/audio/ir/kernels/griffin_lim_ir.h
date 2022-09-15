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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_GRIFFIN_LIM_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_GRIFFIN_LIM_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
namespace audio {
constexpr char kGriffinLimOperation[] = "GriffinLim";

class GriffinLimOperation : public TensorOperation {
 public:
  GriffinLimOperation(int32_t n_fft, int32_t n_iter, int32_t win_length, int32_t hop_length, WindowType window_type,
                      float power, float momentum, int32_t length, bool rand_init);

  ~GriffinLimOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t n_fft_;
  int32_t n_iter_;
  int32_t win_length_;
  int32_t hop_length_;
  WindowType window_type_;
  float power_;
  float momentum_;
  int32_t length_;
  bool rand_init_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_GRIFFIN_LIM_IR_H_
