/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ops/log_uniform_candidate_sampler.h"

#include <memory>
#include <set>
#include <vector>

#include "utils/check_convert_utils.h"
#include "mindspore/core/abstract/ops/op_infer.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void LogUniformCandidateSampler::Init(int64_t num_true, int64_t num_sampled, bool unique, int64_t range_max,
                                      int64_t seed) {
  this->set_num_true(num_true);
  this->set_num_sampled(num_sampled);
  this->set_unique(unique);
  this->set_range_max(range_max);
  this->set_seed(seed);
}

class LogUniformCandidateSamplerInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    int64_t num_sampled = GetValue<int64_t>(primitive->GetAttr(kNumSampled));
    auto sampled_candidate_shape_ptr = std::make_shared<abstract::Shape>(ShapeVector({num_sampled}));
    auto true_expected_shape_ptr = input_args[0]->BuildShape();
    auto true_classes_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(true_expected_shape_ptr)[kShape];
    const size_t true_classes_shape_rank = 2;
    if (!IsDynamicRank(true_classes_shape) && true_classes_shape.size() != true_classes_shape_rank) {
      MS_EXCEPTION(ValueError) << "LogUniformCandidateSampler input true_classes dims should be 2.";
    }
    if (!IsDynamic(true_classes_shape)) {
      auto num_true = GetValue<int64_t>(primitive->GetAttr(kNumTrue));
      if (true_classes_shape[1] != num_true) {
        MS_EXCEPTION(ValueError)
          << "LogUniformCandidateSampler input true_classes dim[1] should equal to num_true, true_classes.dim[1] = "
          << true_classes_shape[1] << ", num_true = " << num_true;
      }
    }

    std::vector<abstract::BaseShapePtr> shape_tuple;
    (void)shape_tuple.emplace_back(sampled_candidate_shape_ptr);
    (void)shape_tuple.emplace_back(true_expected_shape_ptr);
    (void)shape_tuple.emplace_back(sampled_candidate_shape_ptr);
    return std::make_shared<abstract::TupleShape>(shape_tuple);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    if (input_args.empty()) {
      MS_EXCEPTION(ValueError) << "LogUniformCandidateSampler input can not be empty";
    }
    // check input data type
    const std::set<TypePtr> valid_types = {kInt64};
    CheckAndConvertUtils::CheckTensorTypeValid("true_classes", input_args[0]->BuildType(), valid_types,
                                               primitive->name());

    // return outputs data type
    auto sampled_candidate_type = std::make_shared<TensorType>(kInt64);
    auto true_expected_type = std::make_shared<TensorType>(kFloat32);
    auto sampled_expected = std::make_shared<TensorType>(kFloat32);
    return std::make_shared<Tuple>(std::vector<TypePtr>{sampled_candidate_type, true_expected_type, sampled_expected});
  }
};

MIND_API_OPERATOR_IMPL(LogUniformCandidateSampler, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(LogUniformCandidateSampler, prim::kPrimLogUniformCandidateSampler,
                                 LogUniformCandidateSamplerInfer, true);
}  // namespace ops
}  // namespace mindspore
