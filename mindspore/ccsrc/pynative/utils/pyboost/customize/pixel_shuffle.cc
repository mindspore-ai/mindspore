/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/pynative/utils/pyboost/customize/pixel_shuffle.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "mindspore/ops/ops_utils/op_constants.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
template <typename T>
ValueTuplePtr MakeValueTuple(const std::vector<T> &values) {
  std::vector<ValuePtr> elements;
  std::transform(values.begin(), values.end(), std::back_inserter(elements), MakeValue<T>);
  return std::make_shared<ValueTuple>(elements);
}

void CheckPixelShuffleShapes(const std::vector<int64_t> &input_shape, int64_t upscale_factor) {
  if (input_shape.size() < kIndex3) {
    MS_EXCEPTION(ValueError) << "PixelShuffle expects input to have at least 3 dimendions, but got input with "
                             << input_shape.size() << " dimension(s).";
  }
  if (upscale_factor <= 0) {
    MS_EXCEPTION(ValueError) << "PixelShuffle expects a positive upscale_factor, but got " << upscale_factor;
  }
  auto c = input_shape[input_shape.size() - kIndex3];
  auto upscale_factor_squared = upscale_factor * upscale_factor;
  if (c % upscale_factor_squared != 0) {
    MS_EXCEPTION(ValueError) << "PixelShuffle expects its input's 'channel' dimension to"
                             << " be divisible by the square of upscale_factor,"
                             << " but input.shape(-3)=" << c << " is not divisible by " << upscale_factor_squared;
  }
}
}  // namespace
tensor::TensorPtr PixelShuffleCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                        const Int64ImmPtr &upscale_factor_imm) {
  const auto &input_shape = input->shape();
  const auto input_rank = input_shape.size();
  auto upscale_factor = GetValue<int64_t>(upscale_factor_imm);
  CheckPixelShuffleShapes(input_shape, upscale_factor);

  // Format: (B1, ..., Bn), C, H, W
  auto c = input_shape[input_rank - kIndex3];
  auto h = input_shape[input_rank - kIndex2];
  auto w = input_shape[input_rank - kIndex1];
  const auto no_batch_dims = 3;
  const auto input_shape_batch_end = input_shape.end() - no_batch_dims;

  auto upscale_factor_squared = upscale_factor * upscale_factor;
  auto oc = c / upscale_factor_squared;
  auto oh = h * upscale_factor;
  auto ow = w * upscale_factor;

  // First, reshape to split the channels dim from c into 3 separate dims: (oc,
  // upscale_factor, upscale_factor). This allows shuffling to be done next by
  // permuting dims.
  std::vector<int64_t> added_dims_shape(input_shape.begin(), input_shape_batch_end);
  added_dims_shape.insert(added_dims_shape.end(), {oc, upscale_factor, upscale_factor, h, w});
  const auto input_reshaped = reshape(input, added_dims_shape);

  // Next, shuffle by permuting the new upscale_factor dims alongside the height and width dims.
  std::vector<int64_t> permutation(input_shape.begin(), input_shape_batch_end);
  // std::iota is used to maintain the batch dims within the permutation.
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.insert(permutation.end(), {-5,   /* oc */
                                         -2,   /* h */
                                         -4,   /* 1st upscale_factor */
                                         -1,   /* w */
                                         -3}); /* 2nd upscale_factor */
  const auto input_permuted = transpose(input_reshaped, permutation);

  // Finally, upscale by collapsing (h, upscale_factor) -> a single dim (oh)
  // and (w, upscale_factor) -> a single dim (ow).
  std::vector<int64_t> final_shape(input_shape.begin(), input_shape_batch_end);
  final_shape.insert(final_shape.end(), {oc, oh, ow});
  const auto final_out = view(contiguous(input_permuted), final_shape);
  op->set_outputs({final_out});
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
