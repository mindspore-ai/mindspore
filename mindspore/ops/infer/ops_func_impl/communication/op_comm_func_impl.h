/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_OP_COMM_FUNC_IMPL_H
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_OP_COMM_FUNC_IMPL_H

#include <vector>
#include <memory>
#include <string>
#include "ir/tensor.h"
#include "ops/infer_info/infer_info.h"

namespace mindspore {
namespace ops {
uint64_t CheckRankSize(const std::string &name, const std::unique_ptr<InferInfo> &value);
uint64_t GetRankValue(const std::string &name, const std::unique_ptr<InferInfo> &value);
void CheckInferShape(const std::string &name, const ShapeVector &input_shape, const ShapeVector &Soutput_shape);
TypeId CheckInferTypes(const std::string &name, const TypeId type, const TypeId out_type, bool is_reduce_op = false);
TypeId CheckInferType(const std::string &name, const TypeId type);
TypeId CheckReduceInferType(const std::string &name, const TypeId type);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_OP_COMM_FUNC_IMPL_H
