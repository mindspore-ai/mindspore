/**
 * Copyright  2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_OPTIMIZER_INSERT_CAST_TO_PYEXECUTE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_OPTIMIZER_INSERT_CAST_TO_PYEXECUTE_H_

#include <string>
#include <utility>
#include "plugin/device/cpu/optimizer/cpu_pass_utils.h"
#include "include/backend/optimizer/optimizer.h"
#include "ir/anf.h"

namespace mindspore {
namespace opt {
using InsertCastFunction = std::function<AnfNodePtr(const FuncGraphPtr &, const AnfNodePtr &, const std::string &,
                                                    const TypeId &, const TypeId &, const abstract::BaseShapePtr &)>;
class InsertCastToPyExecute : public PatternProcessPass {
 public:
  explicit InsertCastToPyExecute(bool multigraph = true)
      : PatternProcessPass("insert_cast_to_pyexecute", multigraph), insert_cast_function_(AddCastOpNodeToGraph) {}
  explicit InsertCastToPyExecute(InsertCastFunction func, bool multigraph = true)
      : PatternProcessPass("insert_cast_to_pyexecute", multigraph), insert_cast_function_(std::move(func)) {}
  ~InsertCastToPyExecute() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  std::function<AnfNodePtr(const FuncGraphPtr &, const AnfNodePtr &, const std::string &, const TypeId &,
                           const TypeId &, const abstract::BaseShapePtr &)>
    insert_cast_function_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_OPTIMIZER_INSERT_CAST_TO_PYEXECUTE_H_
