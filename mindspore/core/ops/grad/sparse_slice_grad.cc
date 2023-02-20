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
#include "ops/grad/sparse_slice_grad.h"

#include <memory>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(SparseSliceGrad, BaseOperator);
class SparseSliceGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    if (!SparseSliceGradCheckShape(primitive, input_args)) {
      auto y_grad_shape = ShapeVector({abstract::Shape::kShapeDimAny});
      return std::make_shared<abstract::Shape>(y_grad_shape);
    }

    auto indices_shape =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    auto y_grad_shape = ShapeVector({indices_shape[0]});
    return std::make_shared<abstract::Shape>(y_grad_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    const int64_t input_num = 4;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("backprop_val_grad", input_args[kInputIndex0]->BuildType(),
                                                     {kUInt8, kUInt16, kUInt32, kUInt64, kInt8, kInt16, kInt32, kInt64,
                                                      kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool},
                                                     op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", input_args[kInputIndex1]->BuildType(), {kInt64},
                                                     op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("start", input_args[kInputIndex2]->BuildType(), {kInt64}, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("new_indices", input_args[kInputIndex3]->BuildType(), {kInt64},
                                                     op_name);
    auto output_type = input_args[kInputIndex0]->BuildType();
    return output_type;
  }

 private:
  static bool SparseSliceGradCheckShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
    auto op_name = primitive->name();
    auto grad_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto indices_shape_ptr = input_args[kInputIndex1]->BuildShape();
    auto start_shape_ptr = input_args[kInputIndex2]->BuildShape();
    auto new_indices_shape_ptr = input_args[kInputIndex3]->BuildShape();

    auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(grad_shape_ptr)[kShape];
    auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
    auto start_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(start_shape_ptr)[kShape];
    auto new_indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(new_indices_shape_ptr)[kShape];
    if (IsDynamic(grad_shape) || IsDynamic(indices_shape) || IsDynamic(start_shape) || IsDynamic(new_indices_shape)) {
      return false;
    }

    const int64_t indices_rank = 2;
    (void)CheckAndConvertUtils::CheckInteger("rank of backprop_val_grad", SizeToLong(grad_shape.size()), kEqual, 1,
                                             op_name);
    (void)CheckAndConvertUtils::CheckInteger("rank of indices", SizeToLong(indices_shape.size()), kEqual, indices_rank,
                                             op_name);
    (void)CheckAndConvertUtils::CheckInteger("rank of start", SizeToLong(start_shape.size()), kEqual, 1, op_name);
    (void)CheckAndConvertUtils::CheckInteger("rank of new_indices", SizeToLong(new_indices_shape.size()), kEqual,
                                             indices_rank, op_name);

    if (grad_shape[0] != new_indices_shape[0]) {
      MS_EXCEPTION(ValueError)
        << "For SparseSliceGrad, backprop_val_grad.shape[0] must equal to new_indices_shape.shape[0], but "
           "got backprop_val_grad.shape = "
        << grad_shape_ptr->ToString() << " and new_indices.shape = " << new_indices_shape_ptr->ToString() << ".";
    }

    if (indices_shape[1] != start_shape[0]) {
      MS_EXCEPTION(ValueError)
        << "For SparseSliceGrad, indices.shape[1] must equal to start.shape[0], but got indices.shape = "
        << indices_shape_ptr->ToString() << " and start.shape = " << start_shape_ptr->ToString() << ".";
    }

    if (indices_shape[1] != new_indices_shape[1]) {
      MS_EXCEPTION(ValueError)
        << "For SparseSliceGrad, indices.shape[1] must equal to new_indices.shape[1], but got indices.shape = "
        << indices_shape_ptr->ToString() << " and new_indices.shape = " << new_indices_shape_ptr->ToString() << ".";
    }

    return true;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseSliceGrad, prim::kPrimSparseSliceGrad, SparseSliceGradInfer, false);
}  // namespace ops
}  // namespace mindspore
