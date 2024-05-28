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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GENERATOR_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GENERATOR_H_

#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"
#include "mindapi/base/macros.h"

namespace mindspore {
namespace ops {
namespace generator {
constexpr size_t kCmdIndex = 0;
using param_type = int64_t;
using state_type = uint8_t;
const auto ParamType = kInt64;
const auto StateType = kUInt8;
const auto CmdType = kInt64;
const auto ParamTypeId = kNumberTypeInt64;
const auto StateTypeId = kNumberTypeUInt8;
const auto CmdTypeId = kNumberTypeInt64;
enum GeneratorCmd { _START = -1, STEP, SEED, GET_STATE, SET_STATE, UNPACK_STATE, INITIAL_SEED, _END };
}  // namespace generator
class MIND_API GeneratorFuncImpl : public OpFuncImpl {
 public:
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GENERATOR_H_
