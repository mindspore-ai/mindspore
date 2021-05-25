/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/normalize_op.h"

#include <random>
#include <vector>

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
NormalizeOp::NormalizeOp(const std::vector<float> &mean, const std::vector<float> &std) : mean_(mean), std_(std) {
  // pre-calculate normalized mean to be used later in each Compute
  for (int64_t i = 0; i < mean.size(); i++) {
    mean_[i] = mean_[i] / std_[i];
  }
}

Status NormalizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // Doing the Normalization
  return Normalize(input, output, mean_, std_);
}

void NormalizeOp::Print(std::ostream &out) const {
  out << "NormalizeOp, mean: ";
  for (const auto &m : mean_) {
    out << m << ", ";
  }
  out << "}" << std::endl << "std: ";
  for (const auto &s : std_) {
    out << s << ", ";
  }
  out << "}" << std::endl;
}
}  // namespace dataset
}  // namespace mindspore
