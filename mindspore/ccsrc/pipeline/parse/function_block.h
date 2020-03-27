/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef PIPELINE_PARSE_FUNCTION_BLOCK_H_
#define PIPELINE_PARSE_FUNCTION_BLOCK_H_

#include <vector>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <memory>
#include <utility>
#include "pipeline/parse/parse_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parse {

class Parser;
class NameSpace;
class Symbol;
class FunctionBlock;
using FunctionBlockPtr = std::shared_ptr<FunctionBlock>;

// A function block is a straight-line code sequence with no branches, every block has one one exit point
// which is return. When parsing function, loop or branch , we use function block to track the structure of
// the original source code.
class FunctionBlock : public std::enable_shared_from_this<FunctionBlock> {
 public:
  explicit FunctionBlock(const Parser& parser);
  virtual ~FunctionBlock() {}

  FuncGraphPtr func_graph() { return func_graph_; }
  void WriteVariable(const std::string& var_name, const AnfNodePtr& node);
  AnfNodePtr ReadVariable(const std::string& var_name);
  void AddPrevBlock(const FunctionBlockPtr& block);
  void SetPhiArgument(const ParameterPtr& phi);
  void CollectRemovablePhi(const ParameterPtr& phi);
  // A block is matured if all its predecessors is generated
  void Mature();
  CNodePtr ForceToBoolNode(const AnfNodePtr& cond);
  void Jump(const FunctionBlockPtr& block, AnfNodePtr node);
  AnfNodePtr SearchReplaceNode(const std::string& var, const ParameterPtr& phi);
  void ConditionalJump(AnfNodePtr condNode, const FunctionBlockPtr& trueBlock, const FunctionBlockPtr& falseBlock);
  // record the assign statement of self.xx weight parameter ,which will use state_setitem op
  void SetStateAssgin(const AnfNodePtr& target, const std::string& readid);
  void AddAutoDepend(const AnfNodePtr& target);
  void InsertDependItemsBeforeReturn();
  void AddGlobalVar(const std::string& var_name) { (void)global_vars_.insert(var_name); }
  bool IsGlobalVar(const std::string& var_name) { return global_vars_.find(var_name) != global_vars_.end(); }
  AnfNodePtr MakeResolveAstOp(const py::object& op);
  AnfNodePtr MakeResolveClassMember(std::string attr);
  AnfNodePtr MakeResolveSymbol(const std::string& value);
  AnfNodePtr MakeResolveOperation(const std::string& value);
  AnfNodePtr MakeResolve(const std::shared_ptr<NameSpace>& name_space, const std::shared_ptr<Symbol>& resolve_symbol);
  const std::unordered_map<ParameterPtr, AnfNodePtr>& removable_phis() const { return removable_phis_; }

 private:
  // block graph
  FuncGraphPtr func_graph_;

  // the block's parser
  const Parser& parser_;

  // A block is matured if all its prev_blocks is processed
  bool matured_;

  // store the nest-level block
  // refer to comments in Parser::func_block_list_;
  std::vector<FunctionBlock*> prev_blocks_;

  // store args and variable's node
  std::map<std::string, AnfNodePtr> vars_;

  // phi_nodes map the parameter node to variable, it can be resolved if the block's predecessors are processed
  std::map<ParameterPtr, std::string> phi_nodes_;

  // jumps map the successor block and the function call that perform jump
  // refer to comments in Parser::func_block_list_ that how to break the cyclic reference
  std::map<FunctionBlock*, CNodePtr> jumps_;

  // keeps all removable phis which will be removed in one pass.
  std::unordered_map<ParameterPtr, AnfNodePtr> removable_phis_;

  // set state nodes need to insert before function return nodes.
  std::unordered_map<AnfNodePtr, std::string> state_assign_;

  // hold declared global variables in function
  std::set<std::string> global_vars_;

  // other depend need to insert before function return nodes.
  // summary or some other node
  std::vector<AnfNodePtr> auto_depends_;
};

}  // namespace parse
}  // namespace mindspore

#endif  // PIPELINE_PARSE_FUNCTION_BLOCK_H_
