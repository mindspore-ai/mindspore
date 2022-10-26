/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_GET_META_FG_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_GET_META_FG_H

#include <string>
#include <memory>
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
class BpropGetMetaFg : public AnfVisitor {
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsValueNode<MindIRMetaFuncGraph>(node)) {
      return nullptr;
    }
    static mindspore::HashMap<std::string, MetaFuncGraphPtr> meta_fgs{
      {"unpack_call", std::make_shared<prim::UnpackCall>("unpack_call")},
    };
    auto meta_fg_name = GetValueNode<MindIRMetaFuncGraphPtr>(node)->name();
    auto iter = meta_fgs.find(meta_fg_name);
    if (iter == meta_fgs.end()) {
      return nullptr;
    }
    return NewValueNode(iter->second);
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_GET_META_FG_H
