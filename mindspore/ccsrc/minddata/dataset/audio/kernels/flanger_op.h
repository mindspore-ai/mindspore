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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_FLANGER_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_FLANGER_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class FlangerOp : public TensorOp {
 public:
  explicit FlangerOp(int32_t sample_rate, float delay, float depth, float regen, float width, float speed, float phase,
                     Modulation modulation, Interpolation interpolation)
      : sample_rate_(sample_rate),
        delay_(delay),
        depth_(depth),
        regen_(regen),
        width_(width),
        speed_(speed),
        phase_(phase),
        Modulation_(modulation),
        Interpolation_(interpolation) {}

  ~FlangerOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": sample_rate: " << sample_rate_ << ", delay:" << delay_ << ", depth: " << depth_
        << ", regen: " << regen_ << ", width: " << width_ << ", speed: " << speed_ << ", phase: " << phase_
        << ", Modulation: " << static_cast<int>(Modulation_) << ", Interpolation: " << static_cast<int>(Interpolation_)
        << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kFlangerOp; }

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  int32_t sample_rate_;
  float delay_;
  float depth_;
  float regen_;
  float width_;
  float speed_;
  float phase_;
  Modulation Modulation_;
  Interpolation Interpolation_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_FLANGER_OP_H_
