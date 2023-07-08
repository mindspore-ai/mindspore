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

#include "ops/fusion/kv_cache_mgr.h"
#include <set>
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/core/ops/lite_ops.h"

namespace mindspore::ops {
void KVCacheMgr::Init() const {}

MIND_API_OPERATOR_IMPL(KVCacheMgr, BaseOperator);

class MIND_API KVCacheMgrInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto past_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kInputIndex0);
    return past_shape_ptr;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::set<TypePtr> valid_types = {kFloat16};
    const std::set<TypePtr> indices_types = {kInt32};
    auto past_type = input_args[kInputIndex0]->BuildType();
    auto cur_type = input_args[kInputIndex1]->BuildType();
    auto index_type = input_args[kInputIndex2]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("index", index_type, indices_types, primitive->name());
    (void)CheckAndConvertUtils::CheckTensorTypeValid("cur", cur_type, valid_types, primitive->name());
    return CheckAndConvertUtils::CheckTensorTypeValid("past", past_type, valid_types, primitive->name());
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(KVCacheMgr, prim::kPrimKVCacheMgr, KVCacheMgrInfer, false);
}  // namespace mindspore::ops
