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
#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_FFT_ARITHMETIC_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_FFT_ARITHMETIC_H_

#include <vector>
#include <memory>
#include "ir/primitive.h"

namespace mindspore {
namespace ops {
BaseShapePtr FFTInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
BaseShapePtr DCTInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
BaseShapePtr FFTNInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
BaseShapePtr DCTNInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);

TypePtr FFTInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
TypePtr DCTInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);

void FFTCheckInputShape(const PrimitivePtr &primitive, std::vector<int64_t> x_shape_vec, int64_t x_rank);
int32_t FFTCheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
int32_t DCTCheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
int32_t FFTNCheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
int32_t DCTNCheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_FFT_ARITHMETIC_H_
