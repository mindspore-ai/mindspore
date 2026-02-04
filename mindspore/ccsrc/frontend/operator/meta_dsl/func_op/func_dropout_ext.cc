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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/func_dropout_ext.h"
#include "ir/dtype/type.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "primitive/array_ops.h"
#include "primitive/comparison_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_z.h"

namespace mindspore::prim {
void CheckFuncDropoutExtInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  auto p_opt = GetScalarValue<float>(input_args[kIndex1]->GetValue());
  if (p_opt.has_value()) {
    auto p_value = static_cast<double>(p_opt.value());
    if (p_value < 0 || p_value > 1) {
      MS_EXCEPTION(ValueError) << "For FuncDropoutExt, 'p' must be in [0, 1], but got " << p_value;
    }
  }
}

BeginFunction(FuncDropoutExt, input, p, training, inplace, seed, offset) {
  auto input_shape = Shape(input);
  auto input_dtype = DTypeId(input);

  auto return_no_zeros = [&]() {
    auto dropout_ext_impl = [&]() {
      auto results = Call(Prim(DropoutExt), input, p, seed, offset);
      auto output = GetItem(results, Value(0));
      Return(output);
    };
    auto inplace_dropout_ext_impl = [&]() {
      auto results = Call(Prim(DropoutExt), input, p, seed, offset);
      auto output = GetItem(results, Value(0));
      auto inplace_output = Call(Prim(InplaceCopy), input, output, Value(false));
      Return(inplace_output);
    };
    auto is_inplace = Equal(inplace, Value(true));
    Return(If(is_inplace, inplace_dropout_ext_impl, dropout_ext_impl));
  };

  auto return_zeros = [&]() {
    auto zeros_output = [&]() {
      auto zeros_tensor = Call(Prim(Zeros), input_shape, input_dtype);
      auto zeros_output = Call(Prim(Mul), input, zeros_tensor);
      Return(zeros_output);
    };
    auto inplace_zeros_output = [&]() {
      auto zeros_tensor = Call(Prim(Zeros), input_shape, input_dtype);
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

  auto p_invalid = [&]() { Return(Raise("ValueError", "For FuncDropoutExt, 'p' must be in [0, 1].")); };
  auto is_p_invalid = Call(Prim(BoolOr), Less(p, Value(0)), Greater(p, Value(1)));
  Return(If(is_p_invalid, p_invalid, p_valid));
}
EndFunction(FuncDropoutExt)
}  // namespace mindspore::prim
