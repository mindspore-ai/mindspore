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

#include <string>
#include <memory>

#include "ops/make_cootensor.h"
#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/anf_utils.h"
#include "utils/shape_utils.h"
#include "abstract/abstract_value.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "ops/core_ops.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
using abstract::AbstractTensor;
using abstract::AbstractTuple;
AbstractBasePtr MakeCOOTensorInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &args_spec_list) {
  // Inputs: two tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kSizeThree);
  auto indices = abstract::CheckArg<AbstractTensor>(op_name, args_spec_list, kIndexZero);
  auto values = abstract::CheckArg<AbstractTensor>(op_name, args_spec_list, kIndexOne);
  auto dense_shape = abstract::CheckArg<AbstractTuple>(op_name, args_spec_list, kIndexTwo);

  auto indices_dtype = indices->element()->BuildType();
  CheckSparseIndicesDtype(indices_dtype, "Indices");

  auto indices_shp = indices->shape()->shape();
  auto values_shp = values->shape()->shape();
  if (IsShapeEmpty(indices_shp) && IsShapeEmpty(values_shp)) {
    MS_LOG(DEBUG) << "Constructing empty COOTensor! Ignore further shape check.";
    std::vector<abstract::AbstractBasePtr> element_list{indices, values, dense_shape};
    return std::make_shared<abstract::AbstractCOOTensor>(element_list);
  }

  for (const auto &elem_type : dense_shape->ElementsType()) {
    if (!elem_type->isa<Int>()) {
      MS_EXCEPTION(TypeError) << "For COOTensor, the element type of `shape` must be Int, but got "
                              << elem_type->ToString();
    }
  }

  // Convert dense_shape from tuple to shapevector(dense_shape_vec)
  auto dense_shape_vec = GetShapeValue(primitive, dense_shape);
  auto dense_shape_value = dense_shape->BuildValue()->cast<ValueTuplePtr>();
  if (!IsDynamic(dense_shape_vec)) {
    MS_EXCEPTION_IF_NULL(dense_shape_value);
    for (auto dense_shape_elem : dense_shape_vec) {
      if (dense_shape_elem <= 0) {
        MS_EXCEPTION(TypeError) << "For COOTensor, the element of `shape` must be positive, but got "
                                << dense_shape_value->ToString();
      }
    }
  }

  if (IsDynamic(indices_shp) || IsDynamic(values_shp)) {
    MS_LOG(DEBUG) << "Dynamic shape in MakeCOOTensor's inputs! Ignore shape check.";
    AbstractBasePtrList element_list{indices, values, dense_shape};
    return std::make_shared<abstract::AbstractCOOTensor>(element_list);
  }

  CheckSparseShape(indices_shp.size(), kSizeTwo, "Indices");
  CheckSparseShape(values_shp.size(), kSizeOne, "Values");

  if (indices_shp[kIndexZero] != values_shp[kIndexZero]) {
    MS_EXCEPTION(ValueError) << "For COOTensor, `indices.shape[" << kIndexZero << "]` must be equal to `values.shape["
                             << kIndexZero << "]`, but got `indices.shape[" << kIndexZero
                             << "]`: " << indices_shp[kIndexZero] << " and `values.shape[" << kIndexZero
                             << "]`: " << values_shp[kIndexZero];
  }
  constexpr int64_t kDimTwo = 2;
  if (indices_shp[kIndexOne] != kDimTwo) {
    MS_EXCEPTION(ValueError) << "For COOTensor, `indices.shape[" << kIndexOne << "]` must be " << kDimTwo << ",but got "
                             << indices_shp[kIndexOne];
  }

  if (!IsDynamicRank(dense_shape_vec) && LongToSize(indices_shp[kIndexOne]) != dense_shape_vec.size()) {
    MS_EXCEPTION(TypeError) << "For COOTensor, `indices.shape[" << indices_shp << "]` must be equal to the second "
                            << "dimension of `indices`: " << dense_shape_vec.size() << " but got "
                            << indices_shp[kIndexOne];
  }
  AbstractBasePtrList element_list{indices, values, dense_shape};
  return std::make_shared<abstract::AbstractCOOTensor>(element_list);
}
MIND_API_OPERATOR_IMPL(MakeCOOTensor, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(MakeCOOTensor, prim::kPrimMakeCOOTensor, MakeCOOTensorInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
