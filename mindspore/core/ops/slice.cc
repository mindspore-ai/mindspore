/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/slice.h"

#include <memory>
#include <set>
#include <vector>
#include <map>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kSliceInputNum = 3;
std::vector<int64_t> InferImplSliceFuncCalInputValue(const PrimitivePtr &primitive, const ValuePtr &input_value) {
  std::vector<int64_t> tmp_input;
  MS_EXCEPTION_IF_NULL(input_value);
  if (input_value->isa<tensor::Tensor>()) {
    tmp_input = CheckAndConvertUtils::CheckTensorIntValue("slice args value", input_value, primitive->name());
  } else if (input_value->isa<ValueTuple>()) {
    tmp_input = CheckAndConvertUtils::CheckTupleInt("slice args value", input_value, primitive->name());
  } else if (input_value->isa<ValueList>()) {
    tmp_input = CheckAndConvertUtils::CheckListInt("slice args value", input_value, primitive->name());
  } else {
    MS_EXCEPTION(TypeError) << "For Slice, the begin and size must be Tuple or List.";
  }

  return tmp_input;
}

abstract::ShapePtr SliceInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_CHECK_FAIL(input_args.size() == kSliceInputNum, "Slice inputs num error");
  auto input_x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto input_begin_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto input_x_shape = input_x_shape_map[kShape];
  auto input_begin_value_ptr = input_args[kInputIndex1]->BuildValue();
  auto input_size_value_ptr = input_args[kInputIndex2]->BuildValue();
  auto input_begin_shape = input_begin_shape_map[kShape];
  (void)CheckAndConvertUtils::CheckInteger("rank of input_x", SizeToLong(input_x_shape.size()), kGreaterThan, 0,
                                           prim_name);

  bool is_inputx_dyn = IsDynamic(input_x_shape);
  ShapeVector out_shape = {};
  if (input_x_shape[0] == 0) {
    MS_EXCEPTION(ValueError) << "For Slice, the input_x must hava value.";
  }

  if (!IsValueKnown(input_begin_value_ptr) && IsValueKnown(input_size_value_ptr)) {
    auto tmp_input = InferImplSliceFuncCalInputValue(primitive, input_size_value_ptr);
    for (size_t i = 0; i < tmp_input.size(); i++) {
      out_shape.push_back(-1);
    }
    return std::make_shared<abstract::Shape>(out_shape);
  }

  if (!IsValueKnown(input_size_value_ptr)) {
    auto arg = input_args[kInputIndex2];
    if (arg->isa<abstract::AbstractTensor>()) {
      auto abs_tensor = arg->cast<abstract::AbstractTensorPtr>();
      auto tensor_shape = abs_tensor->shape()->shape();
      if (tensor_shape.size() != 1) {
        MS_EXCEPTION(ValueError) << "For Slice, the shape of input|begin|size must be equal.";
      }
    }
    out_shape = GetShapeValue(primitive, input_args[kInputIndex2]);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  auto input_begin_value = InferImplSliceFuncCalInputValue(primitive, input_begin_value_ptr);
  auto input_size_value = InferImplSliceFuncCalInputValue(primitive, input_size_value_ptr);
  auto rank = input_x_shape.size();
  if ((!is_inputx_dyn) && ((input_begin_value.size() != rank) || (input_size_value.size() != rank))) {
    MS_EXCEPTION(ValueError) << "For Slice, the shape of input|begin|size must be equal.";
  }
  (void)CheckAndConvertUtils::CheckPositiveVector("input_begin", input_begin_value, prim_name);
  for (size_t i = 0; i < rank; ++i) {
    if (input_x_shape[i] < 0) {
      continue;
    }
    if (input_size_value[i] < -1) {
      MS_EXCEPTION(RuntimeError) << "For Slice, the value in size should not be less than -1, but got "
                                 << input_size_value[i];
    }

    if (input_begin_value[i] + input_size_value[i] > input_x_shape[i]) {
      MS_EXCEPTION(ValueError) << "For Slice, the sum of begin_shape[" << i << "] and size_shape[" << i
                               << "] must be no greater than input_x_shape[" << i << "].";
    }
    if (input_size_value[i] == -1) {
      input_size_value[i] = input_x_shape[i] - input_begin_value[i];
    }
  }
  return std::make_shared<abstract::Shape>(input_size_value);
}

TypePtr SliceInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return CheckAndConvertUtils::CheckSubClass("input_x", input_args[0]->BuildType(), {kTensorType}, primitive->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(Slice, BaseOperator);
AbstractBasePtr SliceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputIndex3, prim_name);
  auto type = SliceInferType(primitive, input_args);
  auto shape = SliceInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

std::vector<int64_t> Slice::get_begin() const {
  auto value_ptr = GetAttr(kBegin);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Slice::get_size() const {
  auto value_ptr = GetAttr(kSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

// AG means auto generated
class MIND_API AGSliceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1, 2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Slice, prim::kPrimSlice, AGSliceInfer, false);
}  // namespace ops
}  // namespace mindspore
