/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_REDUCE_SCATTER_H_
#define MINDSPORE_CORE_OPS_REDUCE_SCATTER_H_

#include <string>
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReduceScatter = "ReduceScatter";
class MS_CORE_API ReduceScatter : public PrimitiveC {
 public:
  ReduceScatter() : PrimitiveC(kNameReduceScatter) { InitIOName({"input_x"}, {"output"}); }
  ~ReduceScatter() = default;
  MS_DECLARE_PARENT(ReduceScatter, PrimitiveC);
  void Init() {}
  void set_group(const std::string &format);
  std::string get_group() const;
  void set_mode(const ReduceMode &mode);
  ReduceMode get_mode() const;
  void set_rank_size(int rank_size);
  int get_rank_size() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_REDUCE_SCATTER_H_
