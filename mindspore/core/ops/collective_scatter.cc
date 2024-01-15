/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/collective_scatter.h"

#include <memory>
#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "mindspore/core/ops/other_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
void CollectiveScatter::set_group(const string &group) {
  std::string g = group;
  (void)this->AddAttr(kGroup, api::MakeValue(g));
}
std::string CollectiveScatter::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<std::string>(value_ptr);
}

void CollectiveScatter::set_rank_size(int rank_size) {
  (void)this->AddAttr(kRankSize, api::MakeValue(static_cast<int64_t>(rank_size)));
}
int CollectiveScatter::get_rank_size() const {
  auto value_ptr = GetAttr(kRankSize);
  return static_cast<int>(GetValue<int64_t>(value_ptr));
}

class CollectiveScatterInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_ERROR_IF_NULL_W_RET_VAL(primitive, std::make_shared<abstract::Shape>());
    auto rank_size_value_ptr = primitive->GetAttr(kRankSize);
    MS_ERROR_IF_NULL_W_RET_VAL(rank_size_value_ptr, std::make_shared<abstract::Shape>());
    auto rank_size = static_cast<int>(GetValue<int64_t>(rank_size_value_ptr));
    if (rank_size == 0) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the 'rank_size' can not be zero, but got "
                               << rank_size;
    }
    auto abstract_shape = input_args[kIndex0]->GetShape();
    MS_ERROR_IF_NULL_W_RET_VAL(abstract_shape, std::make_shared<abstract::Shape>());
    if (abstract_shape->IsDynamic()) {
      return abstract_shape;
    }
    auto shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(abstract_shape)[kShape];
    if (shape.empty() || shape[0] % rank_size != 0) {
      MS_EXCEPTION(ValueError)
        << "the first dimension for 'input_shape' must be divided by 'rank_size', but got input_shape[0]: " << shape[0]
        << ", rank_size: " << rank_size;
    }
    auto out_shape = shape;
    out_shape[0] = static_cast<int64_t>(shape[0] / rank_size);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto dtype = input_args[kIndex0]->GetType();
    const std::set<TypePtr> default_valid_types = {kInt8, kInt32, kFloat16, kFloat32};
    const std::set<TypePtr> gpu_valid_types = {kBool,   kInt8,    kInt32,   kUInt32, kInt64,
                                               kUInt64, kFloat16, kFloat32, kFloat64};
    const std::string input_name = "input";
    auto context_ptr = MsContext::GetInstance();
    auto is_gpu = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
    if (is_gpu) {
      (void)CheckAndConvertUtils::CheckTensorTypeValid(input_name, dtype, gpu_valid_types, primitive->name());
    } else {
      (void)CheckAndConvertUtils::CheckTensorTypeValid(input_name, dtype, default_valid_types, primitive->name());
    }
    return dtype;
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

MIND_API_OPERATOR_IMPL(CollectiveScatter, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(CollectiveScatter, prim::kPrimCollectiveScatter, CollectiveScatterInfer, false);
}  // namespace ops
}  // namespace mindspore
