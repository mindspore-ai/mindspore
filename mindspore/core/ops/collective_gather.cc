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

#include <set>
#include "ops/collective_gather.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "ops/other_ops.h"
#include "abstract/ops/op_infer.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
void CollectiveGather::set_group(const string &group) {
  std::string g = group;
  (void)this->AddAttr(kGroup, api::MakeValue(g));
}
std::string CollectiveGather::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<std::string>(value_ptr);
}

void CollectiveGather::set_rank_size(int rank_size) {
  (void)this->AddAttr(kRankSize, api::MakeValue(static_cast<int64_t>(rank_size)));
}
int CollectiveGather::get_rank_size() const {
  auto value_ptr = GetAttr(kRankSize);
  return static_cast<int>(GetValue<int64_t>(value_ptr));
}

void CollectiveGather::set_dest_rank(int dest_rank) {
  (void)this->AddAttr(kDestRank, api::MakeValue(static_cast<int64_t>(dest_rank)));
}
int CollectiveGather::get_dest_rank() const {
  auto value_ptr = GetAttr(kDestRank);
  return static_cast<int>(GetValue<int64_t>(value_ptr));
}

class CollectiveGatherInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto x = CheckAndConvertUtils::CheckArgsType(prim_name, input_args, 0, kObjectTypeTensorType);
    MS_EXCEPTION_IF_NULL(x);
    if (!primitive->HasAttr(kRankSize)) {
      MS_LOG(EXCEPTION) << "CollectiveGather doesn't have rank_size attr.";
    }
    auto rank_size_ptr = primitive->GetAttr(kRankSize);
    auto dest_rank_ptr = primitive->GetAttr(kDestRank);
    auto rank_id_ptr = primitive->GetAttr(kRankId);
    auto rank_size = GetValue<int64_t>(rank_size_ptr);
    auto dest_rank = GetValue<int64_t>(dest_rank_ptr);
    auto rank_id = GetValue<int64_t>(rank_id_ptr);
    MS_LOG(INFO) << "For '" << prim_name << "', input rank_id : " << rank_id << ".";
    MS_LOG(INFO) << "For '" << prim_name << "', input rank_size : " << rank_size << ".";
    MS_LOG(INFO) << "For '" << prim_name << "', input dest_rank : " << dest_rank << ".";
    MS_LOG(INFO) << "For '" << prim_name << "', x->shape()->shape()[0] : " << x->GetShape()->GetShapeVector()[0] << ".";
    if (rank_size <= 0) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input rank_size must > 0, but got: " << rank_size << ".";
    }
    if (dest_rank != rank_id) {
      ShapeVector output;
      output.push_back(1);
      return std::make_shared<abstract::Shape>(output);
    }

    auto x_shape = x->GetShape()->GetShapeVector();
    int64_t ret_shape_0;
    if (x_shape[0] >= 1) {
      ret_shape_0 = x_shape[0] * rank_size;
    } else if (x_shape[0] == -1) {
      ret_shape_0 = -1;
    } else {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input x_shape[0] is valid: " << x_shape[0] << ".";
    }
    ShapeVector output;
    output.push_back(ret_shape_0);
    if (x_shape.size() > 1) {
      for (size_t i = 1; i < x_shape.size(); i++) {
        output.push_back(x_shape[i]);
      }
    }
    return std::make_shared<abstract::Shape>(output);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t input_num = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto x_type = input_args[0]->GetType();
    MS_EXCEPTION_IF_NULL(x_type);
    if (!x_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << x_type->ToString()
                              << ".";
    }

    // The data type check is only migrated from the previous corresponding python code,
    // and need further confirmation is required
    const std::set<TypePtr> default_target_dtypes = {kInt8, kUInt8, kInt32, kFloat16, kFloat32};
    const std::set<TypePtr> target_dtypes = common_valid_types_with_bool;
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
    if (!is_ascend) {
      (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, target_dtypes, prim_name);
    } else {
      (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, default_target_dtypes, prim_name);
    }

    return x_type;
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    auto type = InferType(primitive, input_args);
    auto shape = InferShape(primitive, input_args);
    return abstract::MakeAbstract(shape, type);
  }
};

MIND_API_OPERATOR_IMPL(CollectiveGather, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(CollectiveGather, prim::kPrimCollectiveGather, CollectiveGatherInfer, false);

}  // namespace ops
}  // namespace mindspore
