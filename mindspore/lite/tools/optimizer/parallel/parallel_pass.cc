/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/parallel/parallel_pass.h"
#include "include/errorcode.h"
#include "ir/tensor.h"

namespace mindspore {
namespace opt {
bool ParallelPass::IsParallelCareNode(const AnfNodePtr &node) {
  return std::any_of(PARALLEL_LIST.begin(), PARALLEL_LIST.end(), [this, &node](auto &prim) {
    if (CheckPrimitiveType(node, prim)) {
      type_name_ = PrimToString(prim);
      return true;
    } else {
      return false;
    }
  });
}

std::string ParallelPass::PrimToString(const PrimitivePtr &prim) {
  if (type_string.find(prim->name()) == type_string.end()) {
    MS_LOG(EXCEPTION) << "String of the type not registered";
  }
  return type_string.at(prim->name());
}

AnfNodePtr ParallelPass::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    return nullptr;
  }
  if (!utils::isa<CNode>(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(cnode) != lite::RET_OK) {
    return nullptr;
  }
  if (!IsParallelCareNode(node)) {
    return nullptr;
  }
  std::string cnode_name = cnode->fullname_with_scope();
  std::string name = cnode_name;
  std::string orig_name = cnode_name;
  // find operator name first, then operator type name.
  if (split_strategys_.find(name) == split_strategys_.end()) {
    name = type_name_;
  }
  if (cnode_name.find(PARALLEL_NAME_SUFFIX) != std::string::npos) {
    MS_LOG(DEBUG) << " : Skip splited cnode " << cnode_name;
    return nullptr;
  }
  MS_LOG(DEBUG) << " : Reached a parallel care node: " << cnode_name;
  if (split_strategys_.find(name) == split_strategys_.end()) {
    MS_LOG(DEBUG) << name << " : No split strategy for the current CNode.";
    return nullptr;
  }
  cnode->set_fullname_with_scope(cnode_name + PARALLEL_NAME_SUFFIX);
  OperatorInfoPtr operator_ = OperatorInstance(type_name_, orig_name, split_strategys_[name]);
  if (operator_ == nullptr) {
    MS_LOG(EXCEPTION) << "Failure: Create " << name << " OperatorInstance failed";
  }
  operator_->set_cnode(cnode);
  operator_->set_func_graph(func_graph);
  operator_->setFmk(FmkType_);
  if (operator_->Init() == RET_ERROR) {
    MS_LOG(EXCEPTION) << "Failure: operator " << name << " init failed";
  }
  return operator_->replace_op();
}

}  // namespace opt
}  // namespace mindspore
