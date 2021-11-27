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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_TIME_STRETCH_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_TIME_STRETCH_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {

class TimeStretchOp : public TensorOp {
 public:
  /// Default value
  static const float kHopLength;
  static const int kNFreq;
  static const float kFixedRate;

  explicit TimeStretchOp(float hop_length = kHopLength, int n_freq = kNFreq, float fixed_rate = kFixedRate)
      : hop_length_(hop_length), n_freq_(n_freq), fixed_rate_(fixed_rate) {}

  ~TimeStretchOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kTimeStretchOp; }

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

 private:
  float hop_length_;
  int n_freq_;
  float fixed_rate_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_TIME_STRETCH_OP_H_
