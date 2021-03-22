/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_FUNCTION_BLOCK_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_FUNCTION_BLOCK_H_

#include <vector>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <memory>
#include <utility>
#include "pipeline/jit/parse/parse_base.h"
#include "utils/log_adapter.h"
#include "utils/ordered_set.h"

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
  explicit FunctionBlock(const Parser &parser);
  virtual ~FunctionBlock() {}

  FuncGraphPtr func_graph() { return func_graph_; }
  void WriteVariable(const std::string &var_name, const AnfNodePtr &node);
  AnfNodePtr ReadVariable(const std::string &var_name);
  void AddPrevBlock(const FunctionBlockPtr &block);
  void SetPhiArgument(const ParameterPtr &phi);
  bool CollectRemovablePhi(const ParameterPtr &phi);
  // A block is matured if all its predecessors is generated
  void Mature();
  CNodePtr ForceToBoolNode(const AnfNodePtr &cond);
  CNodePtr ForceToWhileCond(const AnfNodePtr &cond);
  void Jump(const FunctionBlockPtr &block, const AnfNodePtr &node);
  AnfNodePtr SearchReplaceNode(const std::string &var, const ParameterPtr &phi);
  void ConditionalJump(AnfNodePtr condNode, const FunctionBlockPtr &trueBlock, const FunctionBlockPtr &falseBlock,
                       bool unroll_loop = true);
  // Create cnode for the assign statement like self.target = source.
  void SetStateAssign(const AnfNodePtr &target, const AnfNodePtr &source);
  void AddGlobalVar(const std::string &var_name) { (void)global_vars_.insert(var_name); }
  bool IsGlobalVar(const std::string &var_name) { return global_vars_.find(var_name) != global_vars_.end(); }
  AnfNodePtr MakeResolveAstOp(const py::object &op);
  AnfNodePtr MakeResolveClassMember(const std::string &attr);
  AnfNodePtr MakeResolveSymbol(const std::string &value);
  AnfNodePtr MakeResolveOperation(const std::string &value);
  AnfNodePtr MakeResolve(const std::shared_ptr<NameSpace> &name_space, const std::shared_ptr<Symbol> &resolve_symbol);
  const std::unordered_map<ParameterPtr, AnfNodePtr> &removable_phis() const { return removable_phis_; }
  void FindIsolatedNodes();
  void AddIsolatedNode(const AnfNodePtr &target);
  void AttachIsolatedNodesBeforeReturn();

 private:
  // Block graph
  FuncGraphPtr func_graph_;

  // Block parser
  const Parser &parser_;

  // A block is matured if all its prev_blocks is processed
  bool matured_;

  // Store the nest-level block.
  // Refer to comments in Parser::func_block_list_;
  std::vector<FunctionBlock *> prev_blocks_;

  // Store args and variable's node, use a bool flag to indicate if the variable is used.
  std::map<std::string, std::pair<AnfNodePtr, bool>> vars_;

  // Map the parameter node to variable, it can be resolved if the block's predecessors are processed
  std::map<ParameterPtr, std::string> phi_nodes_;

  // Jumps map the successor block and the function call that perform jump
  // Refer to comments in Parser::func_block_list_ that how to break the cyclic reference
  std::map<FunctionBlock *, CNodePtr> jumps_;

  // Keep all removable phis which will be removed in one pass.
  std::unordered_map<ParameterPtr, AnfNodePtr> removable_phis_;

  // Keep the map for the resolve node to the removable phi node.
  // For the case that ReadVariable returns a phi node although this phi node
  // generated in the prev block is identified as removable. The other blocks
  // should find this phi node.
  std::unordered_map<AnfNodePtr, ParameterPtr> resolve_to_removable_phis_;

  // Hold declared global variables in function
  std::set<std::string> global_vars_;

  // Keep new made resolve symbol for the variable not found in vars_.
  std::unordered_map<std::string, AnfNodePtr> var_to_resolve_;

  // Isolated nodes.
  OrderedSet<AnfNodePtr> isolated_nodes_;
};

}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_FUNCTION_BLOCK_H_
