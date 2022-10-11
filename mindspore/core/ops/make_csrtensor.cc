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
#include <algorithm>
#include <memory>

#include "ops/make_csrtensor.h"

#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/anf_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
using abstract::AbstractTensor;
using abstract::AbstractTuple;
AbstractBasePtr MakeCSRTensorInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &args_spec_list) {
  // Inputs: three tensors and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kSizeFour);
  auto indptr = abstract::CheckArg<AbstractTensor>(op_name, args_spec_list, kIndexZero);
  auto indices = abstract::CheckArg<AbstractTensor>(op_name, args_spec_list, kIndexOne);
  auto values = abstract::CheckArg<AbstractTensor>(op_name, args_spec_list, kIndexTwo);
  auto shape = abstract::CheckArg<AbstractTuple>(op_name, args_spec_list, kIndexThree);

  auto indptr_dtype = indptr->element()->BuildType();
  auto indices_dtype = indices->element()->BuildType();
  CheckSparseIndicesDtype(indptr_dtype, "indptr");
  CheckSparseIndicesDtype(indices_dtype, "indices");

  auto indptr_shp = indptr->shape()->shape();
  CheckSparseShape(indptr_shp.size(), kSizeOne, "Indptr");

  auto indices_shp = indices->shape()->shape();
  CheckSparseShape(indices_shp.size(), kSizeOne, "Indices");

  auto values_shp = values->shape()->shape();

  // convert shape from tuple to shapevector(shape_vec)
  auto shape_value = shape->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(shape_value);
  auto shp = shape_value->value();
  ShapeVector shape_vec;
  (void)std::transform(std::begin(shp), std::end(shp), std::back_inserter(shape_vec), [](const ValuePtr &e) -> int64_t {
    auto elem = GetValue<int64_t>(e);
    return elem;
  });

  if (values_shp.size() + 1 != shape_vec.size()) {
    MS_EXCEPTION(ValueError) << "Values' dimension should equal to CSRTensor's dimension - 1, but got"
                             << "Values' dimension: " << values_shp.size()
                             << ", CSRTensor's dimension: " << shape_vec.size() << ".";
  }

  if (IsDynamicShape(indptr_shp) || IsDynamicShape(indices_shp) || IsDynamicShape(values_shp) ||
      IsDynamicShape(shape_vec)) {
    MS_LOG(DEBUG) << "Dynamic shape in MakeCSRTensor's inputs! Ignore shape check.";
    std::vector<abstract::AbstractBasePtr> element_list{indptr, indices, values, shape};
    return std::make_shared<abstract::AbstractCSRTensor>(element_list);
  }

  if (indices_shp[kIndexZero] != values_shp[kIndexZero]) {
    MS_EXCEPTION(ValueError) << "Indices and values must have same size, but got: values length: "
                             << values_shp[kIndexZero] << ", indices length " << indices_shp[kIndexZero];
  }

  if (shape_vec[kIndexZero] + 1 != indptr_shp[kIndexZero]) {
    MS_EXCEPTION(ValueError) << "Indptr must have length (1 + shape[0]), but got: " << indptr_shp[kIndexZero];
  }

  size_t shape_size = 1;
  auto shape_types = shape->ElementsType();
  for (size_t i = 0; i < shape_vec.size(); ++i) {
    if (shape_vec[i] <= 0) {
      MS_EXCEPTION(ValueError) << "The element of shape must be positive, but got " << shape_value->ToString();
    }
    if ((i > 1) && (shape_vec[i] != values_shp[i - 1])) {
      MS_EXCEPTION(ValueError)
        << "CSRTensor's shape[2: ] must be equal to value's shape[1: ], but CSRTensor's shape got: "
        << shape_value->ToString() << ", "
        << "values's shape got: " << values->shape()->ToString() << ".";
    }
    if (!shape_types[i]->isa<Int>()) {
      MS_EXCEPTION(TypeError) << "The element type of shape must be Int, but got " << shape_types[i]->ToString();
    }
    shape_size *= LongToSize(shape_vec[i]);
  }
  if (static_cast<int64_t>(shape_size) < values_shp[kIndexZero]) {
    MS_EXCEPTION(ValueError) << "Shape total size: " << shape_size << " is too small to hold " << values_shp[kIndexZero]
                             << " non-zero values.";
  }
  std::vector<abstract::AbstractBasePtr> element_list{indptr, indices, values, shape};
  return std::make_shared<abstract::AbstractCSRTensor>(element_list);
}
MIND_API_OPERATOR_IMPL(MakeCSRTensor, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(MakeCSRTensor, prim::kPrimMakeCSRTensor, MakeCSRTensorInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
