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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/dropout2d_ext.h"
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "primitive/array_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/comparison_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"
#include "primitive/auto_generate/gen_ops_primitive_o.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_z.h"

namespace mindspore::prim {
void CheckDropout2dExtInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  auto p_opt = GetScalarValue<float>(input_args[kIndex1]->GetValue());
  if (p_opt.has_value()) {
    auto p_value = static_cast<double>(p_opt.value());
    if (p_value < 0 || p_value > 1) {
      MS_EXCEPTION(ValueError) << "For Dropout2dExt, 'p' must be in [0, 1], but got " << p_value;
    }
  }

  const auto &input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (!IsDynamicRank(input_shape) && (input_shape.size() == kDim3 || input_shape.size() == kDim4)) {
    MS_LOG(WARNING) << "For Dropout2dExt, " << input_shape.size() << "-D input which is not recommended. "
                    << "Please use Dropout instead.";
  }
}

BeginFunction(Dropout2dExt, input, p, training, inplace, seed, offset) {
  auto input_shape = Shape(input);
  auto input_rank = Rank(input);
  auto input_dtype = DTypeId(input);

  auto rank_valid = [&]() {
    auto dynamic_noise_shape = [&]() {
      auto slice_shape = Call(Prim(SequenceSlice), input_shape, Value(0), Value(2), Value(1));
      auto ones_shape = Call(Prim(TensorToTuple), Call(Prim(OnesLike), Call(Prim(TensorShape), input)));
      auto rest_shape = Call(Prim(SequenceSlice), ones_shape, Value(2), Call(Prim(SequenceLen), ones_shape), Value(1));
      auto noise_shape = Call(Prim(SequenceAdd), slice_shape, rest_shape);
      Return(noise_shape);
    };

    auto static_noise_shape = [&]() {
      auto shape_loop = [&](const NodePtr &index, const NodePtr &item, const NodePtr &result) {
        auto is_reduce_index = Greater(index, Value(1));
        auto reduce_dim = [&]() { Return(Value(1)); };
        auto origin_dim = [&]() { Return(item); };
        auto dim = If(is_reduce_index, reduce_dim, origin_dim);
        auto out = Call(Prim(ListAppend), result, dim);
        Return(out);
      };
      auto list_shape = Call(Prim(TupleToList), input_shape);
      auto tmp_noise_shape = List();
      auto noise_shape = Call(Prim(ListToTuple), For(shape_loop, list_shape, tmp_noise_shape, Value(0), input_rank));
      Return(noise_shape);
    };

    auto is_dynamic = Call(std::make_shared<Primitive>("IsShapeUnKnown"), input_shape);
    auto noise_shape = If(is_dynamic, dynamic_noise_shape, static_noise_shape);
    auto empty = Call(Prim(NewEmpty), input, noise_shape, Value(kNone), Value(kNone));
    auto noise_p = Call(Prim(ScalarSub), Value<float>(1.0), p);
    auto bernoulli_out = Call(Prim(InplaceBernoulliScalar), empty, noise_p, seed, offset);
    auto noise = Call(Prim(InplaceDivs), bernoulli_out, noise_p);

    auto return_output = [&]() {
      auto output = Call(Prim(Mul), input, noise);
      Return(output);
    };
    auto return_inplace_output = [&]() {
      auto inplace_output = Call(Prim(InplaceMul), input, noise);
      Return(inplace_output);
    };

    auto is_inplace = Equal(inplace, Value(true));
    Return(If(is_inplace, return_inplace_output, return_output));
  };

  auto return_no_zeros = [&]() {
    auto rank_invalid = [&]() {
      Return(Raise("ValueError", "For Dropout2dExt, the dimensions of input can't be less than 2."));
    };
    auto is_rank_invalid = Less(input_rank, Value(2));
    Return(If(is_rank_invalid, rank_invalid, rank_valid));
  };

  auto return_zeros = [&]() {
    auto zeros_output = [&]() {
      auto zeros_tensor = Call(Prim(Zeros), Value<std::vector<int64_t>>({}), input_dtype);
      auto zeros_output = Call(Prim(Mul), input, zeros_tensor);
      Return(zeros_output);
    };
    auto inplace_zeros_output = [&]() {
      auto zeros_tensor = Call(Prim(Zeros), Value<std::vector<int64_t>>({}), input_dtype);
      auto inplace_zeros_output = Call(Prim(InplaceMul), input, zeros_tensor);
      Return(inplace_zeros_output);
    };
    auto is_inplace = Equal(inplace, Value(true));
    Return(If(is_inplace, inplace_zeros_output, zeros_output));
  };

  auto p_valid = [&]() {
    auto need_compute = [&]() {
      auto is_output_zeros = Equal(p, Value(1));
      Return(If(is_output_zeros, return_zeros, return_no_zeros));
    };
    auto no_need_compute = [&]() { Return(input); };
    auto is_no_need_args = Call(Prim(BoolOr), Equal(p, Value(0)), Equal(training, Value(false)));
    auto is_no_need_compute = Call(Prim(BoolOr), is_no_need_args, Equal(Call(Prim(Size), input), Value(0)));
    Return(If(is_no_need_compute, no_need_compute, need_compute));
  };

  auto p_invalid = [&]() { Return(Raise("ValueError", "For Dropout2dExt, 'p' must be in [0, 1].")); };
  auto is_p_invalid = Call(Prim(BoolOr), Less(p, Value(0)), Greater(p, Value(1)));
  Return(If(is_p_invalid, p_invalid, p_valid));
}
EndFunction(Dropout2dExt)
}  // namespace mindspore::prim
