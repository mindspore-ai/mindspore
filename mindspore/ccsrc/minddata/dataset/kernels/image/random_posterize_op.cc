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

#include "minddata/dataset/kernels/image/random_posterize_op.h"

#include <opencv2/imgcodecs.hpp>

#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {

const std::vector<uint8_t> RandomPosterizeOp::kBitRange = {4, 8};

RandomPosterizeOp::RandomPosterizeOp(const std::vector<uint8_t> &bit_range)
    : PosterizeOp(bit_range[0]), bit_range_(bit_range) {
  rnd_.seed(GetSeed());
  is_deterministic_ = false;
}

Status RandomPosterizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  CHECK_FAIL_RETURN_UNEXPECTED(input != nullptr, "RandomPosterizeOp: parameter input is nullptr");
  bit_ = (bit_range_[0] == bit_range_[1]) ? bit_range_[0]
                                          : std::uniform_int_distribution<uint8_t>(bit_range_[0], bit_range_[1])(rnd_);
  return PosterizeOp::Compute(input, output);
}
}  // namespace dataset
}  // namespace mindspore
