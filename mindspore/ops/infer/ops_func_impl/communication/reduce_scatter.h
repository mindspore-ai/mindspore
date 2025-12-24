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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <set>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReduceScatter = "ReduceScatter";
class OPS_API ReduceScatter : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ReduceScatter);
  ReduceScatter() : BaseOperator(kNameReduceScatter) { InitIOName({"input_x"}, {"output"}); }
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
