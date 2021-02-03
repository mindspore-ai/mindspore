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

#ifndef MINDSPORE_CORE_OPS_ARG_MIN_H_
#define MINDSPORE_CORE_OPS_ARG_MIN_H_
#include <string>
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameArgMin = "ArgMin";
class ArgMin : public PrimitiveC {
 public:
  ArgMin() : PrimitiveC(kNameArgMin) { InitIOName({"x"}, {"output"}); }
  explicit ArgMin(const std::string k_name) : PrimitiveC(k_name) { InitIOName({"x"}, {"output"}); }
  ~ArgMin() = default;
  MS_DECLARE_PARENT(ArgMin, PrimitiveC);
  void Init(const int64_t axis = -1, const TypeId output_type = kNumberTypeInt32);
  void set_axis(const int64_t axis);
  void set_output_type(const TypeId output_type);

  int64_t get_axis() const;
  TypeId get_output_type() const;
};
AbstractBasePtr ArgMinInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
using PrimArgMin = std::shared_ptr<ArgMin>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ARG_MIN_H_
