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

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t mask_convert_len = 128;
ShapeVector CalDynamicOutputShape(const ValuePtrList value_list) {
  int64_t count = 1;
  size_t x_rank = value_list.size();
  for (std::size_t i = 0; i < x_rank; ++i) {
    auto indexed_value = value_list[i];
    int64_t value = 0;
    if (indexed_value->isa<Int64Imm>()) {
      value = GetValue<int64_t>(indexed_value);
    } else {
      MS_LOG(EXCEPTION) << "DropOutGenMask shape value must be int64, but " << indexed_value->ToString();
    }

    if (value <= 0) {
      MS_LOG(EXCEPTION) << "DropOutGenMask product of value should be > 0";
    }

    if (std::numeric_limits<int64_t>::max() / count / value < 1) {
      MS_LOG(EXCEPTION) << "integer multiply integer overflow";
    }
    count = count * value;
  }

  // convert to bytes(8 bits) mask, using round up
  int64_t n128s = count / mask_convert_len;
  if ((count % mask_convert_len) != 0) {
    n128s++;
  }
  int64_t bytes_count = n128s * 16;

  std::vector<int64_t> shape{bytes_count};
  return shape;
}

ShapeVector CalOutputShape(const AbstractBasePtrList shape_list) {
  int64_t count = 1;
  size_t x_rank = shape_list.size();
  for (std::size_t i = 0; i < x_rank; ++i) {
    auto value_track = shape_list[i]->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);
    int64_t value = 0;
    if (value_track->isa<Int64Imm>()) {
      value = GetValue<int64_t>(value_track);
    } else {
      MS_LOG(EXCEPTION) << "DropoutGenMask input x_shape elements is not int64 or int32, but "
                        << value_track->ToString() << ".";
    }

    if (value <= 0) {
      MS_LOG(EXCEPTION) << "DropOutGenMask product of value should be > 0";
    }

    if (std::numeric_limits<int64_t>::max() / count / value < 1) {
      MS_LOG(EXCEPTION) << "integer multiply integer overflow";
    }
    count = count * value;
  }

  // convert to bytes(8 bits) mask, using round up
  int64_t n128s = count / mask_convert_len;
  if ((count % mask_convert_len) != 0) {
    n128s++;
  }
  int64_t bytes_count = n128s * 16;

  std::vector<int64_t> shape{bytes_count};
  return shape;
}

abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer shape", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           op_name);
  AbstractBasePtr shape_args = input_args[0];
  MS_EXCEPTION_IF_NULL(shape_args);

  if (shape_args->isa<abstract::AbstractTensor>()) {
    auto shape_abstract = dyn_cast<abstract::AbstractTensor>(shape_args);
    MS_EXCEPTION_IF_NULL(shape_abstract);
    auto shape_base = shape_abstract->BuildShape();
    MS_EXCEPTION_IF_NULL(shape_base);
    auto shape = shape_base->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->shape().size() != 1) {
      MS_EXCEPTION(TypeError) << "Input `shape` must be a 1-D Tensor.";
    }
    size_t shape_rank = LongToSize(shape->shape()[0]);

    auto shape_max = shape_abstract->get_max_value();
    MS_EXCEPTION_IF_NULL(shape_max);
    auto max_value = shape_max->isa<ValueList>() ? shape_max->cast<ValueListPtr>()->value()
                                                 : shape_max->cast<ValueTuplePtr>()->value();

    auto shape_min = shape_abstract->get_min_value();
    MS_EXCEPTION_IF_NULL(shape_min);
    auto min_value = shape_min->isa<ValueList>() ? shape_min->cast<ValueListPtr>()->value()
                                                 : shape_min->cast<ValueTuplePtr>()->value();
    if (max_value.size() != shape_rank || min_value.size() != shape_rank) {
      MS_LOG(EXCEPTION) << "The size of max_value or min_value is not equal to the shape rank.";
    }
    ShapeVector out_min_shape = CalDynamicOutputShape(min_value);
    ShapeVector out_max_shape = CalDynamicOutputShape(max_value);
    ShapeVector any_shape{abstract::Shape::SHP_ANY};

    return std::make_shared<abstract::Shape>(any_shape, out_min_shape, out_max_shape);
  }

  auto x_shape = dyn_cast<abstract::AbstractTuple>(shape_args);
  auto x_shape_data = x_shape->elements();
  ShapeVector out_shape = CalOutputShape(x_shape_data);
  return std::make_shared<abstract::Shape>(out_shape);
}
TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("inputs", input_args[1]->BuildType(), valid_types, op_name);
  return kUInt8;
}
}  // namespace

AbstractBasePtr DropoutGenMaskInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(InferShape(primitive, input_args), InferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(DropoutGenMask, prim::kPrimDropoutGenMask, DropoutGenMaskInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
