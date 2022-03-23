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
#include "abstract/primitive_infer_map.h"
#include "mindapi/base/shared_ptr.h"
#include "./op_name.h"

namespace mindspore::ops {
const std::set<TypePtr> common_valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                              kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};

const std::set<TypePtr> common_valid_types_with_complex = {kInt8,    kInt16,     kInt32,     kInt64,   kUInt8,
                                                           kUInt16,  kUInt32,    kUInt64,    kFloat16, kFloat32,
                                                           kFloat64, kComplex64, kComplex128};

const std::set<TypePtr> common_valid_types_with_complex_and_bool = {
  kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,     kUInt32,
  kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool};

const std::set<TypePtr> all_types = {
  kBool,   kInt,    kInt8,   kInt16, kInt32,   kInt64,   kUInt,    kUInt8,
  kUInt16, kUInt32, kUInt64, kFloat, kFloat16, kFloat32, kFloat64, kComplex64,
};
std::vector<int64_t> CalBroadCastShape(std::vector<int64_t> x_shape, std::vector<int64_t> y_shape,
                                       const std::string &op_name, const std::string &op_x_name = "input1",
                                       const std::string &op_y_name = "input2");
abstract::ShapePtr BroadCastInferShape(const std::string &op_name,
                                       const std::vector<abstract::AbstractBasePtr> &input_args);

template <typename T>
api::SharedPtr<T> GetOperator(const AnfNodePtr &node) {
  auto prim = GetValueNode<PrimitivePtr>(node);
  if (prim == nullptr) {
    return nullptr;
  }
  return api::MakeShared<T>(prim);
}
}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_OP_UTILS_H
