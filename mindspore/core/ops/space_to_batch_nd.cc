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
#include "ops/space_to_batch_nd.h"

#include <string>
#include <memory>
#include <set>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
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
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t PADDING_SHAPE_1 = 2;

ShapeVector SpaceToBatchNDInferShapeImpl(const string &kernel_name_, const std::vector<int64_t> &block_size_,
                                         const std::vector<std::vector<int64_t>> &paddings_,
                                         const ShapeVector &input_shape_) {
  if (input_shape_.size() < block_size_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the input size should be no less than the block size, but get input size: "
                      << input_shape_.size() << " block size: " << block_size_.size();
  }

  auto block_rank_ = block_size_.size();
  auto off_set_ = input_shape_.size() - block_size_.size();

  ShapeVector output_shape_ = input_shape_;
  for (size_t i = 0; i < block_rank_; i++) {
    if (block_size_[i] < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the elements of 'block_size' should be both larger than 1, but got " << i
                        << "'th block size " << block_size_[i] << ")\n";
    }
  }

  // check paddings_
  if (paddings_.size() != block_rank_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the size of 'paddings' should be equal to the length of 'block_size':  " << block_rank_
                      << ", but got " << paddings_.size();
  }

  for (size_t idx_i = 0; idx_i < block_rank_; ++idx_i) {
    if (paddings_[idx_i].size() != PADDING_SHAPE_1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the size of each vector of 'paddings' should be equal to the length of 'block_size': "
                        << PADDING_SHAPE_1 << ", but got " << idx_i << "'th element: " << paddings_[idx_i].size();
    }
    for (size_t idx_j = 0; idx_j < PADDING_SHAPE_1; ++idx_j) {
      if (paddings_[idx_i][idx_j] < 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the element of 'paddings' cannot be less than 0, "
                          << "but got paddings[" << idx_i << "][ " << idx_j << "]: " << paddings_[idx_i][idx_j];
      }
    }

    // check the paddings and block_sizes are valid
    auto tmp_shape = input_shape_[idx_i + off_set_] + paddings_[idx_i][0] + paddings_[idx_i][1];
    if (input_shape_[idx_i + off_set_] > 0 && (tmp_shape % block_size_[idx_i]) != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', padded shape should be divisible by block_size, but got padded shape: " << tmp_shape
                        << ", block_size: " << block_size_[idx_i];
    }
    if (input_shape_[idx_i + off_set_] > 0 && (tmp_shape / block_size_[idx_i]) == 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', padded shape cannot be less than block_size"
                        << ", but got padded shape: " << tmp_shape << ", block_size: " << block_size_[idx_i];
    }
    output_shape_[idx_i + off_set_] =
      input_shape_[idx_i + off_set_] > 0 ? tmp_shape / block_size_[idx_i] : input_shape_[idx_i + off_set_];
    output_shape_[0] = output_shape_[0] > 0 ? output_shape_[0] * block_size_[idx_i] : output_shape_[0];
  }

  return output_shape_;
}

abstract::ShapePtr SpaceToBatchNDInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (x_shape.size() != 0 && (IsDynamicRank(x_shape) || x_shape[0] == -1)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  constexpr size_t x_min_len = 2;
  CheckAndConvertUtils::CheckInteger("input_x rank", SizeToLong(x_shape.size()), kGreaterEqual, x_min_len, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);

  auto paddings_value_ptr = primitive->GetAttr(kPaddings);
  MS_EXCEPTION_IF_NULL(paddings_value_ptr);
  auto paddings = GetValue<std::vector<std::vector<int64_t>>>(paddings_value_ptr);

  auto block_shapes_value_ptr = primitive->GetAttr(kBlockShape);
  MS_EXCEPTION_IF_NULL(block_shapes_value_ptr);
  auto block_shapes = GetValue<std::vector<int64_t>>(block_shapes_value_ptr);

  ShapeVector out_shape = SpaceToBatchNDInferShapeImpl(prim_name, block_shapes, paddings, input_shape_ptr->shape());

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr SpaceToBatchNDInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                         kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  auto var_type = input_args[0]->BuildType();

  return CheckAndConvertUtils::CheckTensorTypeValid("input type", var_type, valid_types, prim->name());
}
}  // namespace

void SpaceToBatchND::set_paddings(std::vector<std::vector<int64_t>> paddings) {
  const int64_t pad_size = 2;
  (void)CheckAndConvertUtils::CheckInteger(kPaddings, SizeToLong(paddings.size()), kEqual, pad_size, this->name());
  size_t h = paddings.size();
  size_t w = paddings[0].size();
  std::vector<size_t> temp_w = {2, 2};
  CheckAndConvertUtils::Check(kPaddings, {h, w}, kEqual, temp_w, this->name());
  for (size_t i = 0; i < h; i++) {
    for (size_t j = 0; j < w; j++) {
      (void)CheckAndConvertUtils::CheckInteger(kPaddings, paddings[i][j], kGreaterEqual, 0LL, this->name());
    }
  }
  (void)this->AddAttr(kPaddings, api::MakeValue(paddings));
}

std::vector<std::vector<int64_t>> SpaceToBatchND::get_paddings() const {
  auto value_ptr = GetAttr(kPaddings);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}
void SpaceToBatchND::set_block_shape(std::vector<int64_t> block_shape) {
  const int64_t block_size = 2;
  (void)CheckAndConvertUtils::CheckInteger(kBlockShape, SizeToLong(block_shape.size()), kEqual, block_size,
                                           this->name());
  for (size_t i = 0; i < block_shape.size(); i++) {
    (void)CheckAndConvertUtils::CheckInteger(kBlockShape, block_shape[i], kGreaterEqual, 1LL, this->name());
  }
  (void)this->AddAttr(kBlockShape, api::MakeValue(block_shape));
}

std::vector<int64_t> SpaceToBatchND::get_block_shape() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kBlockShape));
}

abstract::AbstractBasePtr SpaceToBatchNDInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args) {
  {
    return abstract::MakeAbstract(SpaceToBatchNDInferShape(primitive, input_args),
                                  SpaceToBatchNDInferType(primitive, input_args));
  }
}

void SpaceToBatchND::Init(const std::vector<int64_t> block_shape, const std::vector<std::vector<int64_t>> paddings) {
  this->set_paddings(paddings);
  this->set_block_shape(block_shape);
}

MIND_API_OPERATOR_IMPL(SpaceToBatchND, BaseOperator);

// AG means auto generated
class MIND_API AGSpaceToBatchNDInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SpaceToBatchNDInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SpaceToBatchNDInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SpaceToBatchNDInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SpaceToBatchND, prim::kPrimSpaceToBatchND, AGSpaceToBatchNDInfer, false);
}  // namespace ops
}  // namespace mindspore
