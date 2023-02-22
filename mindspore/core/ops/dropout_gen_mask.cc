/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/dropout_gen_mask.h"

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <limits>
#include <algorithm>
#include <cstddef>
#include <iterator>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t kDropoutGenMaskMaskConvertLen = 128;
ShapeVector CalDynamicOutputShape(const PrimitivePtr &primitive, const ValuePtrList value_list) {
  int64_t count = 1;
  size_t x_rank = value_list.size();
  for (std::size_t i = 0; i < x_rank; ++i) {
    auto indexed_value = value_list[i];
    int64_t value = 0;
    if (indexed_value->isa<Int64Imm>()) {
      value = GetValue<int64_t>(indexed_value);
    } else {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', the type of shape value must be int64, but got: " << indexed_value->ToString() << ".";
    }

    if (value <= 0) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', product of value must be greater than 0, but got: " << value << ".";
    }

    if (std::numeric_limits<int64_t>::max() / count / value < 1) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', integer multiply integer overflow.";
    }
    count = count * value;
  }

  // convert to bytes(8 bits) mask, using round up
  int64_t n128s = count / kDropoutGenMaskMaskConvertLen;
  if ((count % kDropoutGenMaskMaskConvertLen) != 0) {
    n128s++;
  }
  int64_t bytes_count = n128s * 16;

  std::vector<int64_t> shape{bytes_count};
  return shape;
}

ShapeVector CalOutputShape(const PrimitivePtr &primitive, const AbstractBasePtrList shape_list) {
  int64_t count = 1;
  size_t x_rank = shape_list.size();
  for (std::size_t i = 0; i < x_rank; ++i) {
    auto value_track = shape_list[i]->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);
    int64_t value = 0;
    if (value_track->isa<Int64Imm>()) {
      value = GetValue<int64_t>(value_track);
    } else {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', input x_shape elements must be int64 or int32, but got: " << value_track->ToString()
                        << ".";
    }

    if (value <= 0) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', product of value must be greater than 0, but got: " << value << ".";
    }

    if (std::numeric_limits<int64_t>::max() / count / value < 1) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', integer multiply integer overflow.";
    }
    count = count * value;
  }
  // convert to bytes(8 bits) mask, using round up
  int64_t n128s = count / kDropoutGenMaskMaskConvertLen;
  if ((count % kDropoutGenMaskMaskConvertLen) != 0) {
    n128s++;
  }
  int64_t bytes_count = n128s * 16;

  std::vector<int64_t> shape{bytes_count};
  return shape;
}

abstract::ShapePtr DropoutGenMaskInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer shape", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           op_name);
  AbstractBasePtr shape_args = input_args[0];
  MS_EXCEPTION_IF_NULL(shape_args);

  ShapeVector out_shape;
  if (shape_args->isa<abstract::AbstractTensor>()) {
    auto shape_value = shape_args->BuildValue();
    MS_EXCEPTION_IF_NULL(shape_value);
    if (shape_value->isa<tensor::Tensor>()) {
      auto mask_shape = CheckAndConvertUtils::CheckTensorIntValue("shape", shape_value, op_name);
      std::vector<ValuePtr> value_elements;
      (void)std::transform(mask_shape.begin(), mask_shape.end(), std::back_inserter(value_elements),
                           [](int64_t elem) { return MakeValue(elem); });
      out_shape = CalDynamicOutputShape(primitive, value_elements);
      return std::make_shared<abstract::Shape>(out_shape);
    }
    auto shape_abstract = dyn_cast<abstract::AbstractTensor>(shape_args);
    MS_EXCEPTION_IF_NULL(shape_abstract);
    auto shape_base = shape_abstract->BuildShape();
    MS_EXCEPTION_IF_NULL(shape_base);
    auto shape = shape_base->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->shape().size() != 1) {
      MS_EXCEPTION(TypeError) << "For '" << op_name
                              << "', input 'shape' must be a 1-D Tensor, but got: " << shape->shape().size() << ".";
    }
    ShapeVector any_shape{abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(any_shape);
  }

  auto shape_value = shape_args->BuildValue();
  MS_EXCEPTION_IF_NULL(shape_value);
  if (!IsValueKnown(shape_value)) {
    ShapeVector any_shape{abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(any_shape);
  }

  auto x_shape = dyn_cast<abstract::AbstractTuple>(shape_args);
  auto x_shape_data = x_shape->elements();
  out_shape = CalOutputShape(primitive, x_shape_data);
  return std::make_shared<abstract::Shape>(out_shape);
}
TypePtr DropoutGenMaskInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer shape", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           op_name);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("inputs", input_args[1]->BuildType(), valid_types, op_name);
  return kUInt8;
}
}  // namespace

MIND_API_OPERATOR_IMPL(DropoutGenMask, BaseOperator);
MIND_API_OPERATOR_IMPL(StatelessDropOutGenMask, DropoutGenMask);
AbstractBasePtr DropoutGenMaskInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return abstract::MakeAbstract(DropoutGenMaskInferShape(primitive, input_args),
                                DropoutGenMaskInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGDropoutGenMaskInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DropoutGenMaskInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DropoutGenMaskInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DropoutGenMaskInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DropoutGenMask, prim::kPrimDropoutGenMask, AGDropoutGenMaskInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(StatelessDropOutGenMask, prim::kPrimStatelessDropOutGenMask, AGDropoutGenMaskInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
