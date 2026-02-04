/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/mla_preprocess.h"
#include <set>
#include <string>
#include <utility>
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "mindapi/helper.h"

namespace mindspore {
namespace ops {
BaseShapePtr MlaPreprocessFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  auto ordinary_input_num = CheckAndConvertUtils::GetRemoveUMonadAbsNum(input_args);
  (void)CheckAndConvertUtils::CheckInteger("inputs num", SizeToLong(ordinary_input_num), kEqual,
                                           kMlaPreProcessInputsNum, op_name);
  auto input1_shape_ptr = input_args[kMlaPreprocessInput1Index]->GetShape();
  auto key_cache_shape_ptr = input_args[kMlaPreprocessKeyCacheIndex]->GetShape();
  auto wuk_ptr = input_args[kMlaPreprocessWukIndex]->GetShape();
  // isDynamicShape
  if (MS_UNLIKELY(IsDynamicRank(input1_shape_ptr->GetShapeVector())) ||
      MS_UNLIKELY(IsDynamicRank(key_cache_shape_ptr->GetShapeVector())) ||
      MS_UNLIKELY(IsDynamicRank(wuk_ptr->GetShapeVector()))) {
    // 不能判断动态rank就返回kShapeRankAny， 会有性能影响
    ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }
  auto cache_mode = GetScalarValue<int64_t>(input_args[kMlaPreprocessParamCacheModeIndex]->GetValue()).value();
  auto head_dim = key_cache_shape_ptr->GetShapeVector()[3];
  auto n = input1_shape_ptr->GetShapeVector()[0];
  auto head_num = wuk_ptr->GetShapeVector()[0];

  ShapeVector output0_shape{n, head_num, head_dim};
  ShapeVector output1_shape{0};
  ShapeVector output2_shape{};
  ShapeVector output3_shape{};
  if (cache_mode != cache_mode_qk_) {
    output0_shape = {n, head_num, 512};
    output1_shape = {0};
    output2_shape = {n, head_num, 64};
    output3_shape = {0};
  }

  auto output0_shape_t = std::make_shared<abstract::TensorShape>(output0_shape);
  auto output1_shape_t = std::make_shared<abstract::TensorShape>(output1_shape);
  auto output2_shape_t = std::make_shared<abstract::TensorShape>(output2_shape);
  auto output3_shape_t = std::make_shared<abstract::TensorShape>(output3_shape);

  return std::make_shared<abstract::TupleShape>(
    abstract::BaseShapePtrList({output0_shape_t, output1_shape_t, output2_shape_t, output3_shape_t}));
}

TypePtr MlaPreprocessFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  TypePtr input1_type = input_args[kMlaPreprocessInput1Index]->GetType();
  TypePtr offset1_type = input_args[kMlaPreprocessQuantOffset1Index]->GetType();

  cache_mode_ = GetScalarValue<int64_t>(input_args[kMlaPreprocessParamCacheModeIndex]->GetValue()).value();
  if (cache_mode_ == cache_mode_qk_split_quant_) {
    return std::make_shared<Tuple>(std::vector<TypePtr>{offset1_type, offset1_type, input1_type, input1_type});
  } else {
    return std::make_shared<Tuple>(std::vector<TypePtr>{input1_type, input1_type, input1_type, input1_type});
  }
}
}  // namespace ops
}  // namespace mindspore
