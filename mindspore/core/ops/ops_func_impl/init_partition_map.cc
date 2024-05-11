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

#include "ops/ops_func_impl/init_partition_map.h"

#include <vector>
#include <string>
#include <set>
#include <memory>

#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
void InitPartitionMapCheckShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto ps_num_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(ps_num_shape);
  if (ps_num_shape->GetShapeVector().size() != 0) {
    MS_LOG(EXCEPTION) << "Dim for ps num must be 0.";
  }
  auto ps_ids_shape = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(ps_ids_shape);
  if (ps_ids_shape->GetShapeVector().size() != 1) {
    MS_LOG(EXCEPTION) << "Dim for ps ids must be 1.";
  }
}

void InitPartitionMapCheckType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto ps_num_shape = input_args[kInputIndex0]->GetType();
  const std::string &op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(ps_num_shape);
  const std::set<TypePtr> valid_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("ps_num", ps_num_shape, valid_types, op_name);
  auto ps_ids_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(ps_ids_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("ps_ids", ps_ids_type, valid_types, op_name);
}
}  // namespace

BaseShapePtr InitPartitionMapFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  InitPartitionMapCheckShape(primitive, input_args);
  return std::make_shared<abstract::TensorShape>(ShapeVector{});
}

TypePtr InitPartitionMapFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  InitPartitionMapCheckType(primitive, input_args);
  return std::make_shared<TensorType>(kInt32);
}
}  // namespace ops
}  // namespace mindspore
