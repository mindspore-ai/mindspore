/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ASCEND_VM_OP_ADAPTER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ASCEND_VM_OP_ADAPTER_H_
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
class AscendVmOpAdapter : public PatternProcessPass {
 public:
  explicit AscendVmOpAdapter(bool multigraph = true) : PatternProcessPass("ascend_vm_op_adapter", multigraph) {}
  ~AscendVmOpAdapter() override = default;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ASCEND_VM_OP_ADAPTER_H_
