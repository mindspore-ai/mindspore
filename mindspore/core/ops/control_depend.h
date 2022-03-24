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

#ifndef MINDSPORE_CORE_OPS_CONTROL_DEPEND_H_
#define MINDSPORE_CORE_OPS_CONTROL_DEPEND_H_
#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameControlDepend = "ControlDepend";
class MIND_API ControlDepend : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ControlDepend);
  ControlDepend() : BaseOperator(kNameControlDepend) {}
  void Init(const int64_t depend_mode);
  void set_depend_mode(const int64_t depend_mode = 0);
  int64_t get_depend_mode() const;
};
using PrimControlDepend = std::shared_ptr<ControlDepend>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_CONTROl_DEPEND_H_
