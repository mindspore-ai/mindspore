/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/solarize_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

Status RandomSolarizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(threshold_min_ <= threshold_max_,
                               "threshold_min must be smaller or equal to threshold_max.");

  uint8_t threshold_min = std::uniform_int_distribution(threshold_min_, threshold_max_)(rnd_);
  uint8_t threshold_max = std::uniform_int_distribution(threshold_min_, threshold_max_)(rnd_);

  if (threshold_max < threshold_min) {
    uint8_t temp = threshold_min;
    threshold_min = threshold_max;
    threshold_max = temp;
  }
  std::unique_ptr<SolarizeOp> op(new SolarizeOp(threshold_min, threshold_max));
  return op->Compute(input, output);
}
}  // namespace dataset
}  // namespace mindspore
