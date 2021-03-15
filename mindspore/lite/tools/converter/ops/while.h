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

#ifndef LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_WHILE_H_
#define LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_WHILE_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

using mindspore::ops::PrimitiveC;

namespace mindspore {
namespace lite {
constexpr auto kNameWhile = "While";
class While : public PrimitiveC {
 public:
  While() : PrimitiveC(kNameWhile) {}
  ~While() = default;
  MS_DECLARE_PARENT(While, PrimitiveC);
  void Init(const int64_t cond_subgraph_index, const int64_t body_subgraph_index);
  void set_cond_subgraph_index(const int64_t cond_subgraph_index);
  void set_body_subgraph_index(const int64_t body_subgraph_index);
  int64_t get_cond_subgraph_index() const;
  int64_t get_body_subgraph_index() const;
};

AbstractBasePtr WhileInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args);
using PrimWhilePtr = std::shared_ptr<While>;
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_WHILE_H_
