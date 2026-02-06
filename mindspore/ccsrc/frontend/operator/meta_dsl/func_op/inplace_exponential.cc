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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/inplace_exponential.h"

#include <set>
#include <limits>

#include "mindapi/base/type_id.h"
#include "ir/dtype/type.h"
#include "ir/dtype/tensor_type.h"
#include "utils/shape_utils.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "primitive/array_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_z.h"
#include "mindspore/core/include/base/float16.h"

namespace mindspore::prim {
namespace {
double GetEps(const TypeId &type_id) {
  const int number_two = 2;
  switch (type_id) {
    case kNumberTypeFloat16:
      return static_cast<double>(std::numeric_limits<float16>::epsilon() / number_two);
    case kNumberTypeBFloat16:
      return static_cast<double>(std::numeric_limits<BFloat16>::epsilon() / number_two);
    case kNumberTypeFloat32:
      return static_cast<double>(std::numeric_limits<float>::epsilon() / number_two);
    default:
      MS_EXCEPTION(ValueError) << "unsupported type_id " << type_id;
  }
}

template <typename T>
void CheckLambdaValue(ValuePtr node) {
  auto val_opt = GetScalarValue<T>(node);
  if (!val_opt.has_value()) {
    return;
  }
  double val = static_cast<double>(val_opt.value());
  if (val <= 0.) {
    MS_EXCEPTION(ValueError) << "exponential_ expects lambda > 0.0, but found lambda=" << val;
  }
}
}  // namespace

void CheckInplaceExponentialInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  auto lambd = input_args[kIndex1]->GetValue();
  auto lambd_type_id = input_args[kIndex1]->GetType()->type_id();
  switch (lambd_type_id) {
    case kNumberTypeInt32:
      CheckLambdaValue<int32_t>(lambd);
      break;
    case kNumberTypeInt64:
      CheckLambdaValue<int64_t>(lambd);
      break;
    case kNumberTypeFloat32:
      CheckLambdaValue<float>(lambd);
      break;
    case kNumberTypeFloat64:
      CheckLambdaValue<double>(lambd);
      break;
    case kNumberTypeBool:
      CheckLambdaValue<bool>(lambd);
      break;
    default:
      MS_EXCEPTION(TypeError) << "Unsupported type: " << lambd->type_name();
  }

  auto input_type_ptr = input_args[kIndex0]->GetType();
  auto input_tensor_type = input_type_ptr->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_tensor_type);
  auto ele = input_tensor_type->element();
  MS_EXCEPTION_IF_NULL(ele);
  auto input_element_type = ele->type_id();
  primitive->AddAttr("input_type_id", MakeValue(static_cast<int64_t>(input_element_type)));
}

BeginFunction(InplaceExponential, input, lambd, seed, offset) {
  auto inf_branch = [&]() {
    auto real_out = Call(Prim(InplaceZero), input);
    Return(real_out);
  };

  auto normal_branch = [&]() {
    const auto &primitive = prim();
    auto input_type_id = primitive->GetAttr("input_type_id");
    MS_EXCEPTION_IF_NULL(input_type_id);
    auto input_type = static_cast<TypeId>(GetValue<int64_t>(input_type_id));
    if (input_type == kNumberTypeFloat64) {
      MS_EXCEPTION(TypeError) << "For InplaceExponential, the float64 input has not been supported.";
    }

    auto uniform_out = Call(Prim(InplaceUniform), input, Value(0.0), Value(1.0), seed, offset);
    auto neg_out = Call(Prim(InplaceMuls), uniform_out, Value(-1.));
    auto add_out = Call(Prim(InplaceAddsExt), neg_out, Value(1.), Value(1.));

    NodePtr real_out = add_out;
    std::set<TypeId> float_types{kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32};
    if (float_types.find(input_type) != float_types.end()) {
      auto eps = GetEps(input_type);
      auto value = Value(1.0 - eps);
      auto mask = Call(Prim(GreaterEqualScalar), add_out, value);
      real_out = Call(Prim(InplaceMaskedFillScalar), add_out, mask, value);
    }

    real_out = Call(Prim(InplaceLog), real_out);
    real_out = Call(Prim(InplaceMuls), real_out, ScalarDiv(Value(-1.), lambd));

    Return(real_out);
  };

  auto double_lambd = Call(Prim(ScalarToTensor), lambd, Value(static_cast<int64_t>(kNumberTypeFloat64)));
  auto condition = Call(Prim(IsInf), double_lambd);

  Return(If(condition, inf_branch, normal_branch));
}
EndFunction(InplaceExponential)

  BeginFunction(InplaceExponentialGrad, input, lambd, seed, offset, out, dout) {
  Return(Tuple(Call(Prim(ZerosLikeExt), input, Value(kNone)), ZerosLike(lambd), ZerosLike(seed), ZerosLike(offset)));
}
EndFunction(InplaceExponentialGrad)
}  // namespace mindspore::prim
