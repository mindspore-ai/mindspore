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

#ifndef MINDSPORE_CONVERT_TUPLE_OUTPUT_TO_MAKETUPLE_H
#define MINDSPORE_CONVERT_TUPLE_OUTPUT_TO_MAKETUPLE_H
#include <string>

#include "ir/anf.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class ConvertTupleOutputToMaketuple : public PatternProcessPass {
 public:
  explicit ConvertTupleOutputToMaketuple(bool multigraph = true)
      : PatternProcessPass("convert_tuple_output_to_maketuple", multigraph) {}

  ~ConvertTupleOutputToMaketuple() override = default;

  const BaseRef DefinePattern() const override;

  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CONVERT_TUPLE_OUTPUT_TO_MAKETUPLE_H
