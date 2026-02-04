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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_MLA_PREPROCESS_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_MLA_PREPROCESS_H_
#include <map>
#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "abstract/abstract_value.h"
#include "primitive/op_name.h"
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
enum MlaPreprocessInputIndex : size_t {
  kMlaPreprocessInput1Index = 0,
  kMlaPreprocessGamma1Index = 1,
  kMlaPreprocessBeta1Index = 2,
  kMlaPreprocessQuantScale1Index = 3,
  kMlaPreprocessQuantOffset1Index = 4,
  kMlaPreprocessWdqkvIndex = 5,
  kMlaPreprocessBias1Index = 6,
  kMlaPreprocessGamma2Index = 7,
  kMlaPreprocessBeta2Index = 8,
  kMlaPreprocessQuantScale2Index = 9,
  kMlaPreprocessQuantOffset2Index = 10,
  kMlaPreprocessGamma3Index = 11,
  kMlaPreprocessSin1Index = 12,
  kMlaPreprocessCos1Index = 13,
  kMlaPreprocessSin2Index = 14,
  kMlaPreprocessCos2Index = 15,
  kMlaPreprocessKeyCacheIndex = 16,
  kMlaPreprocessSlotMappingIndex = 17,
  kMlaPreprocessWuqIndex = 18,
  kMlaPreprocessBias2Index = 19,
  kMlaPreprocessWukIndex = 20,
  kMlaPreprocessDeScale1Index = 21,
  kMlaPreprocessDeScale2Index = 22,
  kMlaPreprocessCtkvScaleIndex = 23,
  kMlaPreprocessQnopeScaleIndex = 24,
  kMlaPreprocessKropeCacheIndex = 25,
  kMlaPreprocessParamCacheModeIndex = 26,
  kMlaPreProcessInputsNum = 27
};

class OPS_API MlaPreprocessFuncImpl : public OpFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

 private:
  mutable int64_t cache_mode_ = 0;
  int64_t cache_mode_qk_ = 0;
  int64_t cache_mode_qk_split_quant_ = 2;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_MLA_PREPROCESS_H_
