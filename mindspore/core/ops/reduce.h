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

#ifndef MINDSPORE_CORE_OPS_REDUCE_H_
#define MINDSPORE_CORE_OPS_REDUCE_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReduce = "Reduce";
class Reduce : public PrimitiveC {
 public:
  Reduce() : PrimitiveC(kNameReduce) { InitIOName({"input_x", "axis"}, {"y"}); }
  explicit Reduce(const std::string k_name) : PrimitiveC(k_name) { InitIOName({"input_x", "axis"}, {"y"}); }
  ~Reduce() = default;
  MS_DECLARE_PARENT(Reduce, PrimitiveC);
  void Init(const bool keep_dims = false);
  void set_keep_dims(const bool keep_dims);
  bool get_keep_dims() const;
};
AbstractBasePtr ReduceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
using PrimReducePtr = std::shared_ptr<Reduce>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REDUCE_H_
