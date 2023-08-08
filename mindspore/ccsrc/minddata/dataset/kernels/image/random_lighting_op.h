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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_LIGHTING_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_LIGHTING_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomLightingOp : public TensorOp {
 public:
  // Default values
  static const float kAlpha;

  explicit RandomLightingOp(float alpha = kAlpha) : dist_(0, alpha) {
    rnd_rgb_.seed(GetSeed());
    is_deterministic_ = false;
  }

  ~RandomLightingOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &in, std::shared_ptr<Tensor> *out) override;

  std::string Name() const override { return kRandomLightingOp; }

 private:
  std::mt19937 rnd_rgb_;
  std::normal_distribution<float> dist_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_RANDOM_LIGHTING_OP_H_
