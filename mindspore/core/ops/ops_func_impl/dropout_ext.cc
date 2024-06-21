/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/dropout_ext.h"
#include <limits>
#include <memory>
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
int64_t CalMaskShape(const PrimitivePtr &primitive, const ShapeVector &shape_vec) {
  constexpr int64_t kDropoutGenMaskMaskConvertLen = 128;
  int64_t count = 1;
  for (size_t i = 0; i < shape_vec.size(); i++) {
    auto dim_value = shape_vec[i];
    if (dim_value <= 0) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', each dim of 'shape' must be greater than 0, but got shape[" << i << "]: " << dim_value
                        << ".";
    }

    if (std::numeric_limits<int64_t>::max() / count / dim_value < 1) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', integer multiply integer overflow.";
    }
    count *= shape_vec[i];
  }

  int64_t n128s = count / kDropoutGenMaskMaskConvertLen;
  if ((count % kDropoutGenMaskMaskConvertLen) != 0) {
    n128s++;
  }
  int64_t bytes_count = n128s * 16;

  return bytes_count;
}

BaseShapePtr DropoutExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_ptr = input_args[kIndex0]->GetShape();
  auto x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  ShapeVector mask_shape;
  if (x_shape_ptr->IsDynamic()) {
    mask_shape.push_back(abstract::TensorShape::kShapeDimAny);
  } else {
    mask_shape.push_back(CalMaskShape(primitive, x_shape));
  }
  auto mask_shape_ptr = std::make_shared<abstract::TensorShape>(mask_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape_ptr, mask_shape_ptr});
}

TypePtr DropoutExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[0]->GetType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, std::make_shared<TensorType>(kUInt8)});
}

int32_t DropoutExtFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex1]);
  const auto &p_opt = GetScalarValue<float>(input_args[kIndex1]->GetValue());
  if (MS_UNLIKELY(!p_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(p_opt.value() >= static_cast<float>(0.0) && p_opt.value() <= static_cast<float>(1.0),
                 "For 'DropoutExt', the 'p' must be in range [0, 1], but got " + std::to_string(p_opt.value()));
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
