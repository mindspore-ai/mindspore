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

#include "ops/vmap_assign.h"

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
// Special handle for empty shape and shape{1}.
inline bool ShapeWithSingleElement(const ShapeVector &shape) {
  return shape.empty() || (shape.size() == 1 && shape[0] == 1);
}

// shape1 is dst_shape, shape2 is source_shape.
bool VmapAssignShapeJoin(const ShapeVector &shape1, const ShapeVector &shape2) {
  if (ShapeWithSingleElement(shape1) && ShapeWithSingleElement(shape2)) {
    return true;
  }
  // shape size not compatible.
  if (shape1.size() != shape2.size()) {
    MS_LOG(ERROR) << "Shape1 size:" << shape1.size() << ", Shape2 size:" << shape2.size();
    return false;
  }
  for (size_t i = 0; i < shape1.size(); ++i) {
    if (shape1[i] == shape2[i]) {
      continue;
    }
    // If shape1 != shape2
    MS_LOG(ERROR) << "Shape1[" << i << "]:" << shape1[i] << ", Shape2[" << i << "]:" << shape2[i] << ".";
    return false;
  }
  return true;
}

std::string GetShapeString(const ShapeVector &tensor_shape) {
  std::ostringstream oss;
  oss << " Shape:";
  for (auto &dim : tensor_shape) {
    oss << " " << dim;
  }
  return oss.str();
}

// The input format is: stacked parameter, param1, param2, ...(a batch of parameters), UMonad.
abstract::ShapePtr VmapAssignInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto stacked_param_shape = dyn_cast_ptr<abstract::Shape>(input_args[0]->BuildShape());
  MS_EXCEPTION_IF_NULL(stacked_param_shape);
  ShapeVector stacked_param_shape_vec = stacked_param_shape->shape();
  if (stacked_param_shape_vec.empty()) {
    MS_EXCEPTION(ValueError) << "stacked_param_shape_vec is empty.";
  }
  auto axis_size = LongToSize(stacked_param_shape_vec[0]);
  if (axis_size != input_args.size() - kNumber2) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the number of inputs excluding stacked parameter "
                             << "and UMonad should be equal to the axis_aize, but get axis_size:" << axis_size
                             << ", and total inputs number:" << input_args.size() << ".";
  }
  (void)stacked_param_shape_vec.erase(stacked_param_shape_vec.begin());
  for (size_t i = 1; i < input_args.size() - 1; ++i) {
    if (!input_args[i]->isa<abstract::AbstractTensor>()) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', input[" << i
                               << "] should be a Tensor, but got:" << input_args[i]->ToString();
    }

    auto shape = dyn_cast_ptr<abstract::Shape>(input_args[i]->BuildShape());
    MS_EXCEPTION_IF_NULL(shape);
    const auto &shape_vec = shape->shape();
    if (!VmapAssignShapeJoin(stacked_param_shape_vec, shape_vec)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', The shape of each parameter in the batch of "
                               << "parameters must be consistent with the shape of the stacked parameter's logical "
                               << "view, but got shape of input[" << i << "]: " << GetShapeString(shape_vec)
                               << ", shape of the stacked parameter's logical view: "
                               << GetShapeString(stacked_param_shape_vec) << ".";
    }
  }
  ShapeVector shape = {1};
  return std::make_shared<abstract::Shape>(shape);
}

TypePtr VmapAssignInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  for (size_t i = 0; i < input_args.size() - 1; ++i) {
    std::string element_i = "element_" + std::to_string(i);
    (void)types.emplace(element_i, input_args[i]->BuildType());
  }
  std::set<TypePtr> valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8, kUInt16,
                                   kUInt32, kUInt64, kFloat16, kFloat32, kBool};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return std::make_shared<TensorType>(kInt32);
}
}  // namespace

MIND_API_OPERATOR_IMPL(VmapStackAssign, BaseOperator);
MIND_API_OPERATOR_IMPL(VmapUnstackAssign, BaseOperator);

AbstractBasePtr VmapAssignInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                           kInputLowerLimit, primitive->name());
  auto infertype = VmapAssignInferType(primitive, input_args);
  auto infershape = VmapAssignInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

// AG means auto generated
class MIND_API AGVmapAssignInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return VmapAssignInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return VmapAssignInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return VmapAssignInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(VmapStackAssign, prim::kPrimVmapStackAssign, AGVmapAssignInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(VmapUnstackAssign, prim::kPrimVmapUnstackAssign, AGVmapAssignInfer, false);
}  // namespace ops
}  // namespace mindspore
