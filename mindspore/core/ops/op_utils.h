/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_OP_UTILS_H
#define MINDSPORE_CORE_OPS_OP_UTILS_H
#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <memory>
#include <climits>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/shape_vector.h"
#include "./op_name.h"

namespace mindspore::ops {
const std::set<TypePtr> common_valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                              kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};

const std::set<TypePtr> common_valid_types_with_bool = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,   kUInt16,
                                                        kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kBool};

const std::set<TypePtr> common_valid_types_with_complex = {kInt8,    kInt16,     kInt32,     kInt64,   kUInt8,
                                                           kUInt16,  kUInt32,    kUInt64,    kFloat16, kFloat32,
                                                           kFloat64, kComplex64, kComplex128};

const std::set<TypePtr> common_valid_types_with_complex_and_bool = {
  kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,     kUInt32,
  kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool};

const std::set<TypePtr> common_float_types = {kFloat16, kFloat32, kFloat64};
const std::set<TypePtr> all_types = {kBool,    kInt,     kInt8,    kInt16,     kInt32,     kInt64,
                                     kUInt,    kUInt8,   kUInt16,  kUInt32,    kUInt64,    kFloat,
                                     kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
std::vector<int64_t> CalBroadCastShape(std::vector<int64_t> x_shape, std::vector<int64_t> y_shape,
                                       const std::string &op_name, const std::string &op_x_name = "input1",
                                       const std::string &op_y_name = "input2");
abstract::ShapePtr BroadCastInferShape(const std::string &op_name,
                                       const std::vector<abstract::AbstractBasePtr> &input_args);
void ReduceFuncCheckAxisInferImpl(const PrimitivePtr &prim, std::vector<int64_t> *axis, const size_t dim);
bool CheckAndGetAxisValue(const std::vector<abstract::AbstractBasePtr> &input_args, std::vector<int64_t> *axis_value,
                          int64_t *axis_shape_v, const PrimitivePtr &primitive);
ShapeVector ReduceFuncCalShapeAxisDyn(const ShapeVector &x_shape, const int64_t axis_shape, bool keep_dims = false);
ShapeVector ReduceFuncCalShapeInferImpl(const PrimitivePtr &primitive, const ShapeVector &x_shape,
                                        const std::vector<int64_t> &axis, bool keep_dims_value = false);
abstract::ShapePtr ReduceBaseInferShape(const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args,
                                        const std::string &prim_name);
TypePtr ReduceBaseInferType(const PrimitivePtr &prim, const std::vector<abstract::AbstractBasePtr> &input_args,
                            const std::set<TypePtr> &check_list);

template <typename T>
api::SharedPtr<T> GetOperator(const AnfNodePtr &node) {
  auto prim = GetValueNode<PrimitivePtr>(node);
  if (prim == nullptr) {
    return nullptr;
  }
  return api::MakeShared<T>(prim);
}

bool ObscureShapeEqual(const ShapeVector &lhs, const ShapeVector &rhs);

// Get the shape value from abstract input arg
// Ops like DynamicBroadcastTo or Reshape can directly get the shape value
// from input which represents shape by invoking this function
// Do not support input with type of AbstractTuple of AbstractTensor
ShapeVector GetShapeValue(const PrimitivePtr &primitive, const AbstractBasePtr &input_arg);

// Infer shape value of make-shape op that only transform shapes, e.g. Concat, Stack, StridedSlice
// Do not support op with multiple outputs for now
ValuePtr InferMakeShapeTensorValue(const PrimitivePtr &prim, const AbstractBasePtrList &args);

// Infer shape value of compute-shape op that could change the dim value, e.g. Mul, Add, Sub
// Do not support op with multiple outputs for now
ValuePtr InferComputeShapeTensorValue(const PrimitivePtr &prim, const AbstractBasePtrList &args);

void CheckSparseShape(ShapeVector sparse_shp, ShapeVector dense_shp);

void CheckSparseShape(const size_t shape_size, const size_t expected_dim, const std::string &arg_name);

void CheckSparseIndicesDtype(const TypePtr data_type, const std::string &arg_name);

void CheckSparseIndicesDtypeInt32(const TypePtr data_type, const std::string &arg_name);

inline void CheckInputShapeEmpty(const std::string &prim_name, const std::vector<AbstractBasePtr> &input_args) {
  for (size_t i = 0; i < input_args.size(); ++i) {
    MS_EXCEPTION_IF_NULL(input_args[i]->BuildShape());
    if (input_args[i]->BuildShape()->IsDimZero()) {
      MS_LOG(EXCEPTION) << "For '" << prim_name << "', input " << i << "'s shape should not be empty!";
    }
  }
}

ShapeVector ConvertToShapeVector(const abstract::AbstractTuplePtr &shape);

template <typename T>
std::shared_ptr<T> InferSparseAttr(const PrimitivePtr &primitive, const AbstractBasePtrList &args_spec_list);

template <typename T>
AbstractBasePtr TensorToSequenceInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);

template <typename T>
AbstractBasePtr InferSequenceSetItem(const PrimitivePtr &primitive, const AbstractBasePtrList &args_spec_list);

template <typename T>
T GetScalarValue(const std::string &op_name, const ValuePtr &elem);

TypePtr HighPriorityType(const TypePtr &x_type, const TypePtr &y_type, const std::string &op_name);

bool IsValueKnown(const ValuePtr &value);

constexpr auto kCSRAvgRows = "csr_avg_rows";
constexpr auto kIsCSR = "is_csr";
constexpr auto kCSRDenseShape = "dense_shape";
constexpr auto kCSRAxis = "axis";
}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_OP_UTILS_H
