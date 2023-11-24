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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_INVERSE_MEL_SCALE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_INVERSE_MEL_SCALE_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "include/dataset/constants.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
class InverseMelScaleOp : public RandomTensorOp {
 public:
  InverseMelScaleOp(int32_t n_stft, int32_t n_mels, int32_t sample_rate, float f_min, float f_max, int32_t max_iter,
                    float tolerance_loss, float tolerance_change, float sgd_lr, float sgd_momentum, NormType norm,
                    MelType mel_type)
      : n_stft_(n_stft),
        n_mels_(n_mels),
        sample_rate_(sample_rate),
        f_min_(f_min),
        f_max_(f_max),
        max_iter_(max_iter),
        tolerance_loss_(tolerance_loss),
        tolerance_change_(tolerance_change),
        sgd_lr_(sgd_lr),
        sgd_momentum_(sgd_momentum),
        norm_(norm),
        mel_type_(mel_type) {}

  ~InverseMelScaleOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kInverseMelScaleOp; }

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  int32_t n_stft_;
  int32_t n_mels_;
  int32_t sample_rate_;
  float f_min_;
  float f_max_;
  int32_t max_iter_;
  float tolerance_loss_;
  float tolerance_change_;
  float sgd_lr_;
  float sgd_momentum_;
  NormType norm_;
  MelType mel_type_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_INVERSE_MEL_SCALE_OP_H_
