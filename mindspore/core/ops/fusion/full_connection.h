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

#ifndef MINDSPORE_CORE_OPS_FULL_CONNECTION_FUSION_H_
#define MINDSPORE_CORE_OPS_FULL_CONNECTION_FUSION_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFullConnection = "FullConnection";
class FullConnection : public PrimitiveC {
 public:
  FullConnection() : PrimitiveC(kNameFullConnection) { InitIOName({"x1", "x2", "b"}, {"output"}); }
  ~FullConnection() = default;
  MS_DECLARE_PARENT(FullConnection, PrimitiveC);
  void Init(const bool has_bias, const int64_t axis, const bool use_axis, const ActivationType &activation_type);
  void set_has_bias(const bool has_bias);
  void set_axis(const int64_t axis);
  void set_use_axis(const bool use_axis);
  void set_activation_type(const ActivationType &activation_type);
  bool get_has_bias() const;
  int64_t get_axis() const;
  bool get_use_axis() const;
  ActivationType get_activation_type() const;
};
AbstractBasePtr FullConnectionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args);
using PrimFullConnectionPtr = std::shared_ptr<FullConnection>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FULL_CONNECTION_FUSION_H_
