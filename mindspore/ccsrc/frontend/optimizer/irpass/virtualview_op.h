/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VIRTUALVIEW_INSERT_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VIRTUALVIEW_INSERT_H_

#include <string>
#include <unordered_map>
#include <utility>
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"

namespace mindspore {
namespace opt {
namespace irpass {
class VirtualViewInsertProcesser {
 public:
  VirtualViewInsertProcesser(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                             bool is_viewed_param_existed)
      : func_graph_(func_graph), manager_(manager), is_viewed_param_existed_(is_viewed_param_existed) {
    params_ = func_graph_->parameters();
  }
  virtual ~VirtualViewInsertProcesser() = default;

  void Run();

 private:
  using ViewDependenceMap = std::unordered_map<AnfNodePtr, AnfNodePtr>;
  using ViewModificationMap = std::unordered_map<AnfNodePtr, std::unordered_map<AnfNodePtr, bool>>;
  using ViewChainMap = std::unordered_map<AnfNodePtr, AnfNodePtrList>;

  static constexpr auto kIsVirtualViewOp = "is_virtual_view_op";
  static constexpr auto kOriginalViewOp = "view_op";

  void InitViewInfoFromParams();
  AnfNodePtr ReplaceWithParameter(const AnfNodePtr &node);
  std::pair<AnfNodePtr, AnfNodePtrList> GetViewInfo(const AnfNodePtr &param);
  AnfNodePtr CreateVirtualViewNode(const AnfNodePtr &view_output, AnfNodePtr *last_umonad);
  void ResetViewModificationStatus(const AnfNodePtr &view_output);
  void VirtualViewInsertAction(const CNodePtr &cnode, const AnfNodePtr &view_node);
  void UpdateViewModificationStatus(const AnfNodePtr &input_node);
  void ProcessViewNode(const CNodePtr &cnode);
  void ProcessInplaceNode(const CNodePtr &cnode);
  void CheckAndProcessInplaceFuncCallNode(const CNodePtr &node);
  void CheckAndInsertVirtualViewOp(const CNodePtr &cnode);
  void ChangeVirtualViewInputInner();
  void DoVirtualViewInputReplace();

  FuncGraphPtr func_graph_;
  FuncGraphManagerPtr manager_;
  bool is_viewed_param_existed_;
  AnfNodePtrList params_;
  std::unordered_map<std::string, AnfNodePtr> refkey_to_param_;
  // m = View(y), n = View(m) -> {m: y, n: y}
  ViewDependenceMap view_dependencies_;
  // m = View(y), n = View(m), Inplace(y) -> {y: {m: true, n: true}}
  ViewModificationMap view_modifications_;
  // m = View(y), n = View(m) -> {m: [m], n: [m, n]}
  ViewChainMap view_chains_;
};

bool VirtualViewInsert(const FuncGraphPtr &root, const opt::OptimizerPtr &opt);
class VirtualViewEliminater : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VIRTUALVIEW_INSERT_H_
