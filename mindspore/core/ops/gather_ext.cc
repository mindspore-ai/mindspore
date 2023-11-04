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

#include "ops/gather_ext.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <set>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/gather_comm.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
abstract::ShapePtr GatherExtInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
 MS_EXCEPTION_IF_NULL(primitive);
 const int64_t input_num = 3;
 const std::string &op_name = primitive->name();
 CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
 abstract::AbstractTensorPtr input =
   CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
 abstract::AbstractTensorPtr index =
   CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 2);
 auto dims = input->shape()->shape().size();
 auto dim = GetValue<int64_t>(input_args[1]->BuildValue());
 if (dim >= static_cast<int64_t>(dims)) {
   MS_EXCEPTION(ValueError)<<"dim:"<<dim <<"greater than input.size:"<<dims;
 }

 return index->shape();
}

TypePtr GatherExtInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
 MS_EXCEPTION_IF_NULL(primitive);
 const std::string &op_name = primitive->name();
 constexpr int64_t input_num = 3;
 CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
 std::set<TypePtr> valid_params_types = {kTensorType};
 (void)CheckAndConvertUtils::CheckSubClass("input", input_args[kInputIndex0]->BuildType(), valid_params_types,
                                           op_name);
 std::set<TypePtr> int_types = {kInt8, kInt16, kInt32, kInt64};
 (void)CheckAndConvertUtils::CheckTensorTypeValid("index", input_args[kInputIndex2]->BuildType(), int_types,
                                                  op_name);
 (void)CheckAndConvertUtils::CheckTypeValid("dim", input_args[kInputIndex1]->BuildType(), int_types, op_name);

 abstract::AbstractTensorPtr input =
   CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
 return input->BuildType();
}

AbstractBasePtr GatherExtInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
 MS_EXCEPTION_IF_NULL(primitive);
 const int64_t kInputsNum = 3;
 CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
 auto infer_type = GatherExtInferType(primitive, input_args);
 auto infer_shape = GatherExtInferShape(primitive, input_args);
 return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(GatherExt, BaseOperator);

// AG means auto generated
class MIND_API AGGatherExtInfer : public abstract::OpInferBase {
public:
 BaseShapePtr InferShape(const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) const override {
   return GatherExtInferShape(primitive, input_args);
 }

 AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const override {
   return GatherExtInfer(engine, primitive, input_args);
 }

 TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
   return GatherExtInferType(primitive, input_args);
 }

 std::set<int64_t> GetValueDependArgIndices() const override { return {2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(GatherExt, prim::kPrimGatherExt, AGGatherExtInfer, false);
}  // namespace ops
}  // namespace mindspore
