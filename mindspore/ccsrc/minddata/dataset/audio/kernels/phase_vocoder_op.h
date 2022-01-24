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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_PHASE_VOCODER_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_PHASE_VOCODER_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class PhaseVocoderOp : public TensorOp {
 public:
  /// \brief Constructor.
  /// \param[in] rate Speed-up factor.
  /// \param[in] phase_advance Expected phase advance in each bin in shape of (freq, 1).
  PhaseVocoderOp(float rate, const std::shared_ptr<Tensor> &phase_advance)
      : rate_(rate), phase_advance_(phase_advance) {}

  ~PhaseVocoderOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kPhaseVocoderOp; }

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  float rate_;
  std::shared_ptr<Tensor> phase_advance_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_PHASE_VOCODER_OP_H_
