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
#include "ops/relu_v2.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ReLUV2, BaseOperator);
class ReLUV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t kReLUV2InputsNum = 1;
    const int64_t input_num = kReLUV2InputsNum;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }

    auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
    auto input_shape = shape_map[kShape];
    if (IsDynamicRank(input_shape)) {
      auto unknow_shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
      return std::make_shared<abstract::TupleShape>(
        std::vector<abstract::BaseShapePtr>{unknow_shape_ptr, unknow_shape_ptr});
    }

    auto x_type_tmp = input_args[0]->BuildType();
    MS_EXCEPTION_IF_NULL(x_type_tmp);
    auto input_type = x_type_tmp->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(input_type);
    auto x_dtype = input_type->element();
    auto mask_shape = ReLUV2GetOutputMaskShape(primitive, input_shape, x_dtype);
    abstract::ShapePtr inputs_shape;
    abstract::ShapePtr masks_shape;
    inputs_shape = std::make_shared<abstract::Shape>(input_shape);
    masks_shape = std::make_shared<abstract::Shape>(mask_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{inputs_shape, masks_shape});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto x_type = input_args[0]->BuildType();
    MS_EXCEPTION_IF_NULL(x_type);
    if (!x_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input type must be tensor, but got: " << x_type->ToString()
                              << ".";
    }
    auto mask_dtype = kUInt8;
    return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, mask_dtype});
  }

 protected:
  std::vector<int64_t> ReLUV2GetOutputMaskShape(const PrimitivePtr &prim, const std::vector<int64_t> &input_shape,
                                                const std::shared_ptr<Type> &x_dtype) const {
    std::vector<int64_t> mask_shape;
    if (input_shape.size() < kInputDims) {
      MS_EXCEPTION(ValueError) << "For '" << prim->name()
                               << "', the dims of 'input_x' must be greater than 4, but got a "
                               << std::to_string(input_shape.size()) << "-D tensor.";
    }
    auto type_id = x_dtype->type_id();
    for (size_t i = 0; i < input_shape.size(); i++) {
      if (i == 1) {
        if (input_shape[1] < 0) {
          mask_shape.push_back(-1);
          continue;
        }
        if (type_id == kNumberTypeInt8 || type_id == kNumberTypeUInt8) {
          mask_shape.push_back(UlongToLong(ceil((input_shape[1] + kFill31) / kRound32)));
        } else {
          mask_shape.push_back(UlongToLong(ceil((input_shape[1] + kFill15) / kRound16)));
        }
      } else {
        mask_shape.push_back(input_shape[i]);
      }
    }
    const int64_t shape_end_4d = 4;
    const int64_t shape_end_2d = 2;
    if (type_id == kNumberTypeInt8 || type_id == kNumberTypeUInt8) {
      (void)mask_shape.insert(mask_shape.end(), shape_end_4d);
    } else {
      (void)mask_shape.insert(mask_shape.end(), shape_end_2d);
    }
    return mask_shape;
  }

 private:
  const size_t kInputDims = 4;
  const int64_t kFill31 = 31;
  const int64_t kRound32 = 32;
  const int64_t kFill15 = 15;
  const int64_t kRound16 = 16;
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ReLUV2, prim::kPrimReLUV2, ReLUV2Infer, false);
}  // namespace ops
}  // namespace mindspore
