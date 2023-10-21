/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_ASCEND_MINDIR_OP_ADAPTER_H
#define MINDSPORE_ASCEND_MINDIR_OP_ADAPTER_H
#include <string>
#include <memory>
#include <map>
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "ir/anf.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/op_adaptation_info_factory.h"

namespace mindspore {
namespace opt {
class AscendMindIROpAdapter : public PatternProcessPass {
 public:
  explicit AscendMindIROpAdapter(bool multigraph = true) : PatternProcessPass("ascend_mindir_op_adapter", multigraph) {}
  ~AscendMindIROpAdapter() override = default;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;
};

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_ASCEND_MINDIR_OP_ADAPTER_H
