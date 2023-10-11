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

#include "ops/index_put.h"

#include <map>
#include <memory>
#include <set>
#include <string>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
void CheckOpValid(const PrimitivePtr &primitive, const ShapeVector &x1_shape, const ShapeVector &x2_shape,
                  const abstract::BaseShapePtrList &idx_shapes, int64_t maxsize) {
  for (size_t idx = 0; idx < idx_shapes.size(); ++idx) {
    auto shape_shape = idx_shapes[idx]->GetShapeVector();
    if (maxsize != shape_shape[0] && shape_shape[0] != 1) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the tensors in indices must be broadcastable, but size of indices[" << idx
                               << "] got " << shape_shape[0] << ".";
    }
  }
  auto accumulate = GetValue<int64_t>(primitive->GetAttr("accumulate"));
  if (accumulate != 0 && accumulate != 1) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', accumulate must be 0 or 1, but got " << accumulate
                             << ".";
  }
  if (idx_shapes.size() > x1_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', rank(x1) must be greater than size(indices), but got " << x1_shape.size() << " vs "
                             << idx_shapes.size() << ".";
  } else if (idx_shapes.size() < x1_shape.size()) {
    if (x2_shape[0] != 1 && x2_shape[0] != x1_shape[x1_shape.size() - 1]) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the size of x2 must be 1 or x1.shape[-1] if rank(x1) > size(indices), but got "
                               << x2_shape[0] << ".";
    }
  } else {
    if (x2_shape[0] != 1 && x2_shape[0] != maxsize) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the size of x2 must be 1 or the max size of the tensors in indices if rank(x1) "
                                  "== size(indices), but got "
                               << x2_shape[0] << ".";
    }
  }
}

abstract::ShapePtr IndexPutInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x1_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kInputIndex0);
  auto x2_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kInputIndex1);
  auto x2_shape = x2_shape_ptr->shape();
  if (IsDynamic(x1_shape_ptr->shape()) || IsDynamic(x2_shape)) {
    return x1_shape_ptr;
  }
  auto x1_shape = x1_shape_ptr->shape();
  auto x2_rank = SizeToLong(x2_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("rank of x2", x2_rank, kEqual, 1, prim_name);
  bool is_tuple_x = input_args[kInputIndex2]->GetType()->object_type() == kObjectTypeTuple;
  bool is_list_x = input_args[kInputIndex2]->GetType()->object_type() == kObjectTypeList;
  if ((!is_tuple_x) && (!is_list_x)) {
    MS_EXCEPTION(TypeError) << "For [" << prim_name << "] should have ListTensor or TupleTensor input but get "
                            << input_args[kInputIndex2]->GetType()->ToString();
  }
  auto idx_shape_ptr = input_args[kInputIndex2]->GetShape();
  MS_EXCEPTION_IF_NULL(idx_shape_ptr);
  size_t idx_size;
  abstract::BaseShapePtrList idx_shapes{};
  if (is_tuple_x) {
    auto shape_tuple = idx_shape_ptr->cast<abstract::TupleShapePtr>();
    idx_shapes = shape_tuple->shape();
    idx_size = shape_tuple->size();
  } else {
    auto shape_list = idx_shape_ptr->cast<abstract::ListShapePtr>();
    idx_shapes = shape_list->shape();
    idx_size = shape_list->size();
  }

  int64_t maxsize = 0;
  for (size_t idx = 0; idx < idx_size; ++idx) {
    auto shape_shape = idx_shapes[idx]->GetShapeVector();
    if (IsDynamic(shape_shape)) {
      return x1_shape_ptr;
    }
    auto idx_rank = SizeToLong(shape_shape.size());
    (void)CheckAndConvertUtils::CheckInteger("rank of indices[" + std::to_string(idx) + "]", idx_rank, kEqual, 1,
                                             prim_name);
    if (maxsize < shape_shape[0]) {
      maxsize = shape_shape[0];
    }
  }
  CheckOpValid(primitive, x1_shape, x2_shape, idx_shapes, maxsize);
  return x1_shape_ptr;
}

TypePtr IndexPutInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kUInt8, kUInt16,    kUInt32,    kUInt64,
                                         kInt8,    kInt16,   kInt32,   kInt64, kComplex64, kComplex128};
  const std::set<TypePtr> idx_valid_types = {kInt32, kInt64};
  auto x1_type = input_args[kInputIndex0]->GetType();
  auto x2_type = input_args[kInputIndex1]->GetType();
  bool is_tuple_x = input_args[kInputIndex2]->GetType()->object_type() == kObjectTypeTuple;
  bool is_list_x = input_args[kInputIndex2]->GetType()->object_type() == kObjectTypeList;
  if ((!is_tuple_x) && (!is_list_x)) {
    MS_EXCEPTION(TypeError) << "For [" << prim_name << "] should have ListTensor or TupleTensor input but get "
                            << input_args[kInputIndex2]->GetType()->ToString();
  }
  auto idx_type = input_args[kInputIndex2]->GetType();
  TypePtrList types_list;
  size_t idx_size;
  if (is_tuple_x) {
    types_list = idx_type->cast<TuplePtr>()->elements();
    idx_size = types_list.size();
  } else {
    types_list = idx_type->cast<ListPtr>()->elements();
    idx_size = types_list.size();
  }
  std::map<std::string, TypePtr> idx_types;
  for (size_t idx = 0; idx < idx_size; ++idx) {
    (void)idx_types.emplace("indices[" + std::to_string(idx) + "]:", types_list[idx]);
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(idx_types, idx_valid_types, prim_name);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x1", x1_type);
  (void)types.emplace("x2", x2_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return x1_type;
}

AbstractBasePtr IndexPutInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputsNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto type = IndexPutInferType(primitive, input_args);
  auto shape = IndexPutInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
}  // namespace

MIND_API_OPERATOR_IMPL(IndexPut, BaseOperator);
class MIND_API AGIndexPutInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return IndexPutInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return IndexPutInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return IndexPutInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(IndexPut, prim::kPrimIndexPut, AGIndexPutInfer, false);
}  // namespace ops
}  // namespace mindspore
