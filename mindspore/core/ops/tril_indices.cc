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

#include "ops/tril_indices.h"

#include <algorithm>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TrilIndicesInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &) {
  auto row_ptr = primitive->GetAttr("row");
  MS_EXCEPTION_IF_NULL(row_ptr);
  auto col_ptr = primitive->GetAttr("col");
  MS_EXCEPTION_IF_NULL(col_ptr);
  auto offset_ptr = primitive->GetAttr("offset");
  MS_EXCEPTION_IF_NULL(offset_ptr);

  int64_t row = GetValue<int64_t>(row_ptr);
  int64_t col = GetValue<int64_t>(col_ptr);
  int64_t offset = GetValue<int64_t>(offset_ptr);
  int64_t tril_size = 0;
  if (row != 0 && col != 0) {
    auto m_first_row = offset > 0 ? std::min<int64_t>(col, 1 + offset) : row + offset > 0;
    auto m_last_row = std::max<int64_t>(0, std::min<int64_t>(col, row + offset));
    auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(row, row + offset));
    auto n_row_trapezoid = (m_last_row - m_first_row + 1);
    tril_size = static_cast<int64_t>(LongToUlong((m_first_row + m_last_row) * n_row_trapezoid) >> 1);
    auto diff_row = n_row_all - n_row_trapezoid;
    if (diff_row > 0) {
      tril_size += diff_row * col;
    }
  }
  ShapeVector y_shape = {2, tril_size};
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr TrilIndicesInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &) {
  auto dtype_attr = primitive->GetAttr("dtype");
  MS_EXCEPTION_IF_NULL(dtype_attr);
  auto infer_type = dtype_attr->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(infer_type);
  return infer_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(TrilIndices, BaseOperator);
void TrilIndices::Init(const int64_t row, const int64_t col, const int64_t offset) {
  set_row(row);
  set_col(col);
  set_offset(offset);
}

void TrilIndices::set_row(const int64_t row) { (void)this->AddAttr(kRow, api::MakeValue(row)); }

void TrilIndices::set_col(const int64_t col) { (void)this->AddAttr(kCol, api::MakeValue(col)); }

void TrilIndices::set_offset(const int64_t offset) { (void)this->AddAttr(kOffset, api::MakeValue(offset)); }

int64_t TrilIndices::get_row() const { return GetValue<int64_t>(GetAttr(kRow)); }

int64_t TrilIndices::get_col() const { return GetValue<int64_t>(GetAttr(kCol)); }

int64_t TrilIndices::get_offset() const { return GetValue<int64_t>(GetAttr(kOffset)); }

AbstractBasePtr TrilIndicesInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 0;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = TrilIndicesInferType(primitive, input_args);
  auto infer_shape = TrilIndicesInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGTrilIndicesInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return TrilIndicesInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return TrilIndicesInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return TrilIndicesInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(TrilIndices, prim::kPrimTrilIndices, AGTrilIndicesInfer, false);
}  // namespace ops
}  // namespace mindspore
