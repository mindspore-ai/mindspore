/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/random_solarize_op.h"

#include <utility>

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status RandomSolarizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  uint8_t threshold_min_ = threshold_[0], threshold_max_ = threshold_[1];

  CHECK_FAIL_RETURN_UNEXPECTED(threshold_min_ <= threshold_max_,
                               "RandomSolarize: min of threshold: " + std::to_string(threshold_min_) +
                                 " is greater than max of threshold: " + std::to_string(threshold_max_));

  float threshold_min = static_cast<float>(
    std::uniform_int_distribution(static_cast<uint32_t>(threshold_min_), static_cast<uint32_t>(threshold_max_))(rnd_));
  float threshold_max = static_cast<float>(
    std::uniform_int_distribution(static_cast<uint32_t>(threshold_min_), static_cast<uint32_t>(threshold_max_))(rnd_));

  if (threshold_max < threshold_min) {
    std::swap(threshold_min, threshold_max);
  }
  return Solarize(input, output, {threshold_min, threshold_max});
}
}  // namespace dataset
}  // namespace mindspore
