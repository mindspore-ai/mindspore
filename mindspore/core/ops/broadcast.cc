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

#include <set>
#include <vector>
#include <memory>
#include "ops/broadcast.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void Broadcast::Init(const int64_t root_rank, const std::string &group) {
  this->set_root_rank(root_rank);
  this->set_group(group);
}
void Broadcast::set_root_rank(const int64_t root_rank) { this->AddAttr(kKeepProb, MakeValue(root_rank)); }

void Broadcast::set_group(const std::string &group) {
  CheckAndConvertUtils::CheckString(kGroup, group, {"hccl_world_group", "hccl_world_group"}, this->name());
  this->AddAttr(kGroup, MakeValue(group));
}
int64_t Broadcast::get_root_rank() const {
  auto value_ptr = this->GetAttr(kRootRank);
  return GetValue<float>(value_ptr);
}

std::string Broadcast::get_group() const {
  auto value_ptr = this->GetAttr(kGroup);
  return GetValue<std::string>(value_ptr);
}
AbstractBasePtr BroadcastInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto broadcast_prim = primitive->cast<PrimBroadcast>();
  MS_EXCEPTION_IF_NULL(broadcast_prim);
  auto prim_name = broadcast_prim->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // infer shape
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  // infer type
  auto x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  std::vector<TypePtr> output_types;
  const std::set<TypeId> valid_types = {kNumberTypeInt8, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat32};
  for (size_t i = 0; i < input_args.size(); i++) {
    auto out_type = input_args[i]->BuildType()->cast<TensorTypePtr>()->element();
    output_types.push_back(out_type);
    CheckAndConvertUtils::CheckTensorTypeValid("index_type", out_type, valid_types, prim_name);
  }
  return std::make_shared<abstract::AbstractTensor>(x_type, in_shape);
}
REGISTER_PRIMITIVE_C(kNameBroadcast, Broadcast);
}  // namespace ops
}  // namespace mindspore
