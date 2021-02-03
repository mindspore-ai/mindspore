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
#ifndef MINDSPORE_CORE_OPS_CUSTOM_PREDICT_H_
#define MINDSPORE_CORE_OPS_CUSTOM_PREDICT_H_
#include <memory>
#include <vector>

#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCustomPredict = "CustomPredict";
class CustomPredict : public PrimitiveC {
 public:
  CustomPredict() : PrimitiveC(kNameCustomPredict) {}
  ~CustomPredict() = default;
  MS_DECLARE_PARENT(CustomPredict, PrimitiveC);
  void Init(const int64_t output_num, const float weight_threshold);
  void set_output_num(const int64_t output_num);
  void set_weight_threshold(const float weight_threshold);
  int64_t get_output_num() const;
  float get_weight_threshold() const;
};
AbstractBasePtr CustomPredictInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args);
using PrimCustomPredictPtr = std::shared_ptr<CustomPredict>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CUSTOM_PREDICT_H_
