/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <functional>
#include <memory>
#include "ops/ops_func_impl/non_zero.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr int64_t kNonZeroInputMinDim = 1;

BaseShapePtr NonZeroFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  const auto &x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();

  MS_CHECK_VALUE(!IsDynamic(x_shape), primitive->name() + "error: shape should not has dynamic values");
  auto x_rank = SizeToLong(x_shape.size());
  MS_CHECK_VALUE(x_rank >= kNonZeroInputMinDim,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("dimension of 'x'", x_rank, kGreaterEqual,
                                                             kNonZeroInputMinDim, primitive));

  auto x_num = std::accumulate(x_shape.begin(), x_shape.end(), int64_t(1), std::multiplies<int64_t>());
  return std::make_shared<abstract::Shape>(ShapeVector({x_num, x_rank}));
}

TypePtr NonZeroFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  std::vector<TypeId> valid_types = {kNumberTypeBool,   kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,
                                     kNumberTypeInt64,  kNumberTypeUInt8,   kNumberTypeUInt16,  kNumberTypeUInt32,
                                     kNumberTypeUInt64, kNumberTypeFloat16, kNumberTypeFloat64, kNumberTypeFloat};
  auto tensor_type = input_args[kInputIndex0]->GetType()->cast<TensorTypePtr>();
  auto real_type = tensor_type->element()->type_id();
  if (std::find(valid_types.begin(), valid_types.end(), real_type) == valid_types.end()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "', input[0] type should be bool, int8, int16, int32, int64, uint8, uint16, uint32, "
                               "uint64, float16, float or float64. but got "
                            << tensor_type->element()->ToString();
  }
  return std::make_shared<TensorType>(kInt64);
}

class NonZeroFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  // Do not override this interface if the op has no InferValue
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    const auto &x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
    auto x_rank = SizeToLong(x_shape.size());
    MS_CHECK_VALUE(x_rank >= kNonZeroInputMinDim,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("dimension of 'x'", x_rank, kGreaterEqual,
                                                               kNonZeroInputMinDim, primitive));
    if (IsDynamicRank(x_shape)) {
      x_rank = abstract::Shape::kShapeDimAny;
    }
    auto output_shape = ShapeVector({abstract::Shape::kShapeDimAny, x_rank});
    return std::make_shared<abstract::AbstractTensor>(kInt64, output_shape);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("NonZero", NonZeroFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
