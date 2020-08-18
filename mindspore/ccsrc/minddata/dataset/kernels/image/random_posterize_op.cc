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

#include <random>
#include <opencv2/imgcodecs.hpp>

#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {

const uint8_t RandomPosterizeOp::kMinBit = 8;
const uint8_t RandomPosterizeOp::kMaxBit = 8;

RandomPosterizeOp::RandomPosterizeOp(uint8_t min_bit, uint8_t max_bit)
    : PosterizeOp(min_bit), min_bit_(min_bit), max_bit_(max_bit) {
  rnd_.seed(GetSeed());
}

Status RandomPosterizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  bit_ = (min_bit_ == max_bit_) ? min_bit_ : std::uniform_int_distribution<uint8_t>(min_bit_, max_bit_)(rnd_);
  return PosterizeOp::Compute(input, output);
}
}  // namespace dataset
}  // namespace mindspore
