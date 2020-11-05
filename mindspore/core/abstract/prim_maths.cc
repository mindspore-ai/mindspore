/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "abstract/infer_functions.h"
#include "abstract/utils.h"
#include "abstract/param_validator.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplMinOrMaxGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: three tensors.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto input_y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto dout = CheckArg<AbstractTensor>(op_name, args_spec_list, 2);
  (void)CheckTensorsDTypeSame({input_x, input_y, dout}, {kInt, kUInt, kFloat},
                              op_name + "evaluator three inputs should be %s");

  AbstractBasePtr dx = input_x->Broaden();
  AbstractBasePtr dy = input_y->Broaden();

  return std::make_shared<AbstractTuple>(AbstractBasePtrList({dx, dy}));
}

AbstractBasePtr InferImplSqrt(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  // Inputs: three tensors.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto inp = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  return inp->Clone()->Broaden();
}

AbstractBasePtr InferImplMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  ShapePtr shape_x = dyn_cast<Shape>(args_spec_list[0]->GetShapeTrack());
  MS_EXCEPTION_IF_NULL(shape_x);
  std::vector<int64_t> x_dims = shape_x->shape();
  ShapePtr shape_y = dyn_cast<Shape>(args_spec_list[1]->GetShapeTrack());
  MS_EXCEPTION_IF_NULL(shape_y);
  std::vector<int64_t> y_dims = shape_y->shape();
  auto broadcast_shape = BroadcastShape(x_dims, y_dims);
  if (broadcast_shape.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_spec_list[0]->ToString() << ","
                      << args_spec_list[1]->ToString();
  }
  auto out = args_spec_list[0]->Broaden();
  out->set_shape(std::make_shared<Shape>(broadcast_shape));
  return out;
}

AbstractBasePtr InferImplTensorAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  ShapePtr shape_x = dyn_cast<Shape>(args_spec_list[0]->GetShapeTrack());
  MS_EXCEPTION_IF_NULL(shape_x);
  std::vector<int64_t> x_dims = shape_x->shape();
  ShapePtr shape_y = dyn_cast<Shape>(args_spec_list[1]->GetShapeTrack());
  MS_EXCEPTION_IF_NULL(shape_y);
  std::vector<int64_t> y_dims = shape_y->shape();
  auto broadcast_shape = BroadcastShape(x_dims, y_dims);
  if (broadcast_shape.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_spec_list[0]->ToString() << ","
                      << args_spec_list[1]->ToString();
  }
  auto out = args_spec_list[0]->Broaden();
  out->set_shape(std::make_shared<Shape>(broadcast_shape));
  return out;
}

AbstractBasePtr InferImplSquare(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: one tensor.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  return args_spec_list[0]->Broaden();
}
}  // namespace abstract
}  // namespace mindspore
