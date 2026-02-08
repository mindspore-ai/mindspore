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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/pixel_shuffle.h"
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "primitive/array_ops.h"
#include "primitive/sequence_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::prim {
void CheckPixelShuffleInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  const auto &input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (!IsDynamicRank(input_shape) && input_shape.size() < kIndex3) {
    MS_EXCEPTION(ValueError) << "PixelShuffle expects input to have at least 3 dimensions, but got input with "
                             << input_shape.size() << " dimension(s).";
  }

  auto upscale_factor_opt = GetScalarValue<int64_t>(input_args[kIndex1]->GetValue());
  if (!upscale_factor_opt.has_value()) {
    return;
  }
  auto upscale_factor = upscale_factor_opt.value();
  if (upscale_factor <= 0) {
    MS_EXCEPTION(ValueError) << "PixelShuffle expects a positive upscale_factor, but got " << upscale_factor;
  }

  auto c = IsDynamicRank(input_shape) ? abstract::Shape::kShapeDimAny : input_shape[input_shape.size() - kIndex3];
  auto upscale_factor_squared = upscale_factor * upscale_factor;
  if (c != abstract::Shape::kShapeDimAny && c % upscale_factor_squared != 0) {
    MS_EXCEPTION(ValueError)
      << "PixelShuffle expects its input's 'channel' dimension to be divisible by the square of upscale_factor, but "
         "input.shape(-3)="
      << c << " is not divisible by " << upscale_factor_squared;
  }
}

BeginFunction(PixelShuffle, input, upscale_factor) {
  // Raise error if dimension is invalid.
  auto invalid_dimension = [&]() {
    Return(Raise("ValueError",
                 "For 'PixelShuffle', the input's 'channel' dimension is not divisible"
                 "by the square of upscale_factor."));
  };
  // Raise error if upscale_factor is not a positive integer.
  auto invalid_upscale_factor = [&]() {
    Return(Raise("ValueError", "For 'PixelShuffle', the 'upscale_factor' must be positive."));
  };

  auto input_shape = Shape(input);
  auto input_rank = Rank(input);
  auto c = GetItem(input_shape, ScalarSub(input_rank, Value(3)));
  auto upscale_factor_squared = ScalarPow(upscale_factor, Value(2));
  // Operator concatenation.
  auto pixel_shuffle_calc = [&]() {
    // Format: (B1, ..., Bn), C, H, W
    auto h = GetItem(input_shape, ScalarSub(input_rank, Value(2)));
    auto w = GetItem(input_shape, ScalarSub(input_rank, Value(1)));

    auto oc = ScalarFloorDiv(c, upscale_factor_squared);
    auto oh = ScalarMul(h, upscale_factor);
    auto ow = ScalarMul(w, upscale_factor);
    auto input_shape_batch_end = ScalarSub(input_rank, Value(3));

    // First, reshape to split the channels dim from c into 3 separate dims: (oc,
    // upscale_factor, upscale_factor). This allows shuffling to be done next by
    // permuting dims.
    auto added_dims_pre = Call(Prim(SequenceSlice), input_shape, Value(0), input_shape_batch_end, Value(1));
    auto added_dims_shape = Call(Prim(SequenceAdd), added_dims_pre, Tuple(oc, upscale_factor, upscale_factor, h, w));
    auto input_reshaped = Reshape(input, added_dims_shape);

    // Next, shuffle by permuting the new upscale_factor dims alongside the height and width dims.
    const auto &end = input_shape_batch_end;
    auto permutation = Call(Prim(MakeRange), end);
    permutation = Call(Prim(SequenceAdd), permutation,
                       Tuple(end, ScalarAdd(end, Value(3)), ScalarAdd(end, Value(1)), ScalarAdd(end, Value(4)),
                             ScalarAdd(end, Value(2))));
    auto input_permuted = Call(Prim(Transpose), input_reshaped, permutation);

    // Finally, upscale by collapsing (h, upscale_factor) -> a single dim (oh)
    // and (w, upscale_factor) -> a single dim (ow).
    auto final_shape = Call(Prim(SequenceAdd), added_dims_pre, Tuple(oc, oh, ow));
    auto out = Reshape(Call(Prim(Contiguous), input_permuted), final_shape);
    Return(out);
  };
  // Implement.
  auto impl_branch = [&]() {
    // the mod of c and upscale_factor_squared, expects to be 0
    auto calc_out = ScalarMod(c, upscale_factor_squared);
    Return(If(NotEqual(calc_out, Value(0)), invalid_dimension, pixel_shuffle_calc));
  };
  // upscalefactor expects to be positive
  Return(If(LessEqual(upscale_factor, Value(0)), invalid_upscale_factor, impl_branch));
}
EndFunction(PixelShuffle)
}  // namespace mindspore::prim
