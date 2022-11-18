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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_META_FG_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_META_FG_H

#include <string>
#include <memory>
#include <utility>
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
class GetMetaFg : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    static mindspore::HashMap<std::string, std::pair<std::string, std::string>> multitype_ops{
      {"zeros_like_leaf", {"zeros_like", ""}},
      {"getitem", {"getitem", "mindspore.ops.composite.multitype_ops.getitem_impl"}},
      {"negative", {"negative", "mindspore.ops.composite.multitype_ops.negative_impl"}},
      {"mul", {"mul", "mindspore.ops.composite.multitype_ops.mul_impl"}},
      {"logical_not", {"logical_not", "mindspore.ops.composite.multitype_ops.logic_not_impl"}},
      {"in", {"in_", "mindspore.ops.composite.multitype_ops.in_impl"}},
      {"less", {"less", "mindspore.ops.composite.multitype_ops.less_impl"}},
      {"less_equal", {"less_equal", "mindspore.ops.composite.multitype_ops.less_equal_impl"}},
      {"greater", {"greater", "mindspore.ops.composite.multitype_ops.greater_impl"}},
      {"add", {"add", "mindspore.ops.composite.multitype_ops.add_impl"}},
      {"sub", {"sub", "mindspore.ops.composite.multitype_ops.sub_impl"}},
      {"dout_cast", {"dout_cast", "mindspore.ops._grad.grad_array_ops"}},
      {"equal", {"equal", "mindspore.ops.composite.multitype_ops.equal_impl"}},
      {"floordiv", {"floordiv", "mindspore.ops.composite.multitype_ops.floordiv_impl"}},
    };
    static mindspore::HashMap<std::string, MetaFuncGraphPtr> meta_fgs{
      {"unpack_call", std::make_shared<prim::UnpackCall>("unpack_call")},
    };
    if (!IsValueNode<MindIRMetaFuncGraph>(node)) {
      return nullptr;
    }
    auto meta_fg_name = GetValueNode<MindIRMetaFuncGraphPtr>(node)->name();
    meta_fg_name = GetHyperMapOpsName(meta_fg_name);
    // meta func_graph
    auto meta_fgs_iter = meta_fgs.find(meta_fg_name);
    if (meta_fgs_iter != meta_fgs.end()) {
      return NewValueNode(meta_fgs_iter->second);
    }
    // multitype func_graph
    auto multitype_ops_iter = multitype_ops.find(meta_fg_name);
    if (multitype_ops_iter == multitype_ops.end()) {
      return nullptr;
    }
    ValuePtr python_ops;
    if (!multitype_ops_iter->second.second.empty()) {
      python_ops = prim::GetPythonOps(multitype_ops_iter->second.first, multitype_ops_iter->second.second);
    } else {
      python_ops = prim::GetPythonOps(multitype_ops_iter->second.first);
    }
    return NewValueNode(python_ops);
  }

  // hyper_map[xxx] -> xxx
  static std::string GetHyperMapOpsName(const std::string &meta_fg_name) {
    static constexpr char kHyperMapPrefix[] = "hyper_map";
    static size_t prefix_len = strlen(kHyperMapPrefix);
    if (meta_fg_name.compare(0, prefix_len, kHyperMapPrefix) != 0) {
      return meta_fg_name;
    }
    constexpr auto offset = 2;
    auto op_name = meta_fg_name.substr(prefix_len + 1, meta_fg_name.length() - prefix_len - offset);
    return op_name;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_META_FG_H
