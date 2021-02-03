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

#ifndef MINDSPORE_CORE_OPS_QUANTD_TYPE_CAST_H_
#define MINDSPORE_CORE_OPS_QUANTD_TYPE_CAST_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "abstract/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameQuantDTypeCast = "QuantDTypeCast";
class QuantDTypeCast : public PrimitiveC {
 public:
  QuantDTypeCast() : PrimitiveC(kNameQuantDTypeCast) {}
  ~QuantDTypeCast() = default;
  MS_DECLARE_PARENT(QuantDTypeCast, PrimitiveC);
  void Init(const int64_t src_t, const int64_t dst_t);
  void set_src_t(const int64_t src_t);
  int64_t get_src_t() const;
  void set_dst_t(const int64_t dst_t);
  int64_t get_dst_t() const;
};
AbstractBasePtr QuantDTypeCastInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args);
using PrimQuantDTypeCastPtr = std::shared_ptr<QuantDTypeCast>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_QUANTD_TYPE_CAST_H_
