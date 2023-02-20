/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/batch_to_space.h"

#include <set>
#include <memory>
#include <string>

#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(BatchToSpace, BaseOperator);

void BatchToSpace::Init(const std::vector<int64_t> &block_size, const std::vector<std::vector<int64_t>> &crops) {
  this->set_block_size(block_size);
  this->set_crops(crops);
}

void BatchToSpace::set_block_size(const std::vector<int64_t> &block_size) {
  (void)this->AddAttr(kBlockSize, api::MakeValue(block_size));
}

std::vector<int64_t> BatchToSpace::get_block_size() const {
  auto value_ptr = this->GetAttr(kBlockSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void BatchToSpace::set_crops(const std::vector<std::vector<int64_t>> &crops) {
  (void)this->AddAttr(kCrops, api::MakeValue(crops));
}

std::vector<std::vector<int64_t>> BatchToSpace::get_crops() const {
  auto value_ptr = this->GetAttr(kCrops);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}

class BatchToSpaceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = 1;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
    auto x_shape = x->BuildShape();
    MS_EXCEPTION_IF_NULL(x_shape);
    auto shape_element = x_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_element);
    auto input_shape = shape_element->shape();
    const size_t input_rank = 4;
    if (input_shape.size() != input_rank) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', rank of 'input_x' should be 4, but got "
                               << shape_element->shape().size();
    }
    if (mindspore::IsDynamicRank(shape_element->shape())) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    }
    auto block_size = GetValue<int64_t>(primitive->GetAttr(kBlockSize));
    auto crops = GetValue<std::vector<std::vector<int64_t>>>(primitive->GetAttr(kCrops));
    const size_t height_dim_index = 2;
    ShapeVector output_shape(input_rank);
    for (size_t i = 0; i < height_dim_index; i++) {
      output_shape[i] = input_shape[i];
    }
    for (size_t i = height_dim_index; i < input_rank; i++) {
      auto x_block_prod = input_shape[i] * block_size;
      auto crop_sum = crops[i - height_dim_index][0] + crops[i - height_dim_index][1];
      if (x_block_prod <= crop_sum) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', prod of 'block_size' and 'input_x' shape should be greater than sum of 'crops',"
                                 << " but got prod of 'block_size' and 'input_x' shape: " << x_block_prod
                                 << ", sum of 'crops': " << crop_sum;
      }
      output_shape[i] = x_block_prod - crop_sum;
    }
    auto block_size_prod = block_size * block_size;
    if (output_shape[0] % block_size_prod != 0) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the shape of output with index 0 must be divided exactly "
                               << "by square of 'block_size', but got the shape of output: " << output_shape
                               << " and square of 'block_size': " << block_size_prod << ".";
    }
    output_shape[0] = output_shape[0] / block_size_prod;
    return std::make_shared<abstract::Shape>(output_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = 1;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,    kUInt32,
                                           kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
    auto x_type = input_args[kInputIndex0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
    return input_args[kInputIndex0]->BuildType();
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchToSpace, prim::kPrimBatchToSpace, BatchToSpaceInfer, false);
}  // namespace ops
}  // namespace mindspore
