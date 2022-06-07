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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_RESAMPLE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_RESAMPLE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class ResampleOp : public TensorOp {
 public:
  ResampleOp(float orig_freq, float new_freq, ResampleMethod resample_method, int32_t lowpass_filter_width,
             float rolloff, float beta)
      : orig_freq_(orig_freq),
        new_freq_(new_freq),
        resample_method_(resample_method),
        lowpass_filter_width_(lowpass_filter_width),
        rolloff_(rolloff),
        beta_(beta) {}

  ~ResampleOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kResampleOp; }

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  float orig_freq_;
  float new_freq_;
  ResampleMethod resample_method_;
  int32_t lowpass_filter_width_;
  float rolloff_;
  float beta_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_RESAMPLE_OP_H_
