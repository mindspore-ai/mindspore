/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEVICE_OPTIMIZER_FORMAT_TYPE_PASS_INSERT_CAST_FOR_RUNOP_H_
#define MINDSPORE_CCSRC_DEVICE_OPTIMIZER_FORMAT_TYPE_PASS_INSERT_CAST_FOR_RUNOP_H_
#include <string>

#include "pre_activate/common/optimizer.h"
#include "pre_activate/common/pattern_engine.h"
#include "ir/anf.h"
namespace mindspore {
namespace opt {
class RunOpInsertCast : public PatternProcessPass {
 public:
  explicit RunOpInsertCast(bool multigraph = true) : PatternProcessPass("insert_cast_for_runop", multigraph) {}
  ~RunOpInsertCast() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_OPTIMIZER_FORMAT_TYPE_PASS_INSERT_CAST_FOR_RUNOP_H_
