/**
 * Copyright 2023~2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_BACKEND_GE_BACKEND_PASS_ASCEND_MINDIR_OP_ADAPTER_H
#define MINDSPORE_BACKEND_GE_BACKEND_PASS_ASCEND_MINDIR_OP_ADAPTER_H
#include <string>
#include <memory>
#include <map>
#include "include/backend/common/pass_manager/optimizer.h"
#include "include/backend/common/pass_manager/op_adaptation_info_factory.h"

namespace mindspore {
namespace opt {
class AscendMindIROpAdapterForGe : public PatternProcessPass {
 public:
  explicit AscendMindIROpAdapterForGe(bool multigraph = true)
      : PatternProcessPass("ascend_mindir_op_adapter_for_ge", multigraph) {}
  ~AscendMindIROpAdapterForGe() override = default;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;
};

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_BACKEND_GE_BACKEND_PASS_ASCEND_MINDIR_OP_ADAPTER_H
