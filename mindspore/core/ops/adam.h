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

#ifndef MINDSPORE_CORE_OPS_ADAM_H_
#define MINDSPORE_CORE_OPS_ADAM_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdam = "Adam";
class Adam : public PrimitiveC {
 public:
  Adam() : PrimitiveC(kNameAdam) {}
  ~Adam() = default;
  MS_DECLARE_PARENT(Adam, PrimitiveC);
  void Init(const bool use_locking = false, const bool use_nesterov = false);
  void set_use_locking(const bool use_locking);
  void set_use_nesterov(const bool use_nesterov);
  bool get_use_locking() const;
  bool get_use_nesterov() const;
};
AbstractBasePtr AdamInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args);
using PrimAdamPtr = std::shared_ptr<Adam>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADAM_H_
