/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <memory>
#include <utility>
#include <tuple>

#include "utils/hash_map.h"
#include "ir/meta_func_graph.h"
#include "pipeline/jit/parse/parse_base.h"
#include "utils/log_adapter.h"
#include "utils/ordered_set.h"

namespace mindspore {
namespace parse {
class Parser;
class NameSpace;
class Symbol;
class Script;
class FunctionBlock;
using FunctionBlockPtr = std::shared_ptr<FunctionBlock>;

// A function block is a straight-line code sequence with no branches, every block has one one exit point
// which is return. When parsing function, loop or branch , we use function block to track the structure of
// the original source code.
class FunctionBlock : public std::enable_shared_from_this<FunctionBlock> {
 public:
  explicit FunctionBlock(const Parser &parser);
  virtual ~FunctionBlock() = default;

  FuncGraphPtr func_graph() { return func_graph_; }
  std::string ToString() const { return func_graph_->ToString(); }
  void WriteVariable(const std::string &var_name, const AnfNodePtr &node);
  AnfNodePtr ReadVariable(const std::string &var_name);
  void AddPrevBlock(const FunctionBlockPtr &block);
  void SetPhiArgument(const ParameterPtr &phi);
  void CollectRemovablePhi(const ParameterPtr &phi);
  // A block is matured if all its predecessors is generated
  void Mature();
  CNodePtr ForceToBoolNode(const AnfNodePtr &cond);
  CNodePtr ForceToWhileCond(const AnfNodePtr &cond);
  void Jump(const FunctionBlockPtr &target_block, const std::vector<AnfNodePtr> &args);
  std::set<AnfNodePtr> SearchAllArgsOfPhiNode(const std::string &var, const ParameterPtr &phi);
  CNodePtr ConditionalJump(const AnfNodePtr &cond_node, const AnfNodePtr &true_block_call,
                           const AnfNodePtr &false_block_call);
  CNodePtr ConditionalJump(const AnfNodePtr &cond_node, const FunctionBlockPtr &true_block,
                           const FunctionBlockPtr &false_block);
  // Create cnode for the assign statement like self.target = source.
  void SetStateAssign(const AnfNodePtr &target, const AnfNodePtr &source);
  void AddGlobalVar(const std::string &var_name) { (void)global_vars_.insert(var_name); }
  bool IsGlobalVar(const std::string &var_name) { return global_vars_.find(var_name) != global_vars_.end(); }
  AnfNodePtr MakeResolveAstOp(const py::object &op);
  AnfNodePtr MakeResolveClassMember(const std::string &attr);
  AnfNodePtr MakeResolveSymbol(const std::string &value);
  AnfNodePtr MakeResolveOperation(const std::string &value);
  AnfNodePtr MakeResolve(const std::shared_ptr<NameSpace> &name_space, const std::shared_ptr<Symbol> &resolve_symbol);
  AnfNodePtr GetResolveNode(const py::tuple &info);
  AnfNodePtr HandleNamespaceInfo(const py::tuple &info);
  AnfNodePtr HandleBuiltinNamespaceInfo(const py::tuple &info);
  AnfNodePtr MakeInterpret(const std::string &script_text, const AnfNodePtr &global_dict_node,
                           const AnfNodePtr &local_dict_node, const AnfNodePtr &orig_node);
  const std::map<ParameterPtr, std::set<AnfNodePtr>> &phi_args() const { return phi_args_; }
  void FindIsolatedNodes();
  void AddIsolatedNode(const AnfNodePtr &target);
  void AttachIsolatedNodesBeforeReturn();
  const std::vector<FunctionBlock *> &prev_blocks() const { return prev_blocks_; }
  bool is_dead_block() const { return is_dead_block_; }
  void SetAsDeadBlock();
  CNodePtr GetJumpNode(FunctionBlock *target_block);

  bool is_return_statement_inside() const { return is_return_statement_inside_; }
  void SetReturnStatementInside();
  bool is_break_continue_statement_inside() const { return is_break_continue_statement_inside_; }
  void SetBreakContinueStatementInside();

  const py::dict &global_py_params() const { return global_py_params_; }
  void set_global_py_params(const py::dict &symbols) { global_py_params_ = symbols; }
  void AddGlobalPyParam(const std::string &name, const py::object &obj) {
    MS_LOG(DEBUG) << "Add global param '" << name << "', " << py::str(obj) << " for the block:" << ToString();
    global_py_params_[py::str(name)] = obj;
  }
  void UpdateGlobalPyParam(const py::dict &symbols) {
    for (auto &param : symbols) {
      if (!global_py_params_.contains(param.first)) {
        MS_LOG(DEBUG) << "Update global param '" << param.first << "', " << py::str(param.second)
                      << " for the block:" << ToString();
        global_py_params_[param.first] = param.second;
      }
    }
  }

  std::tuple<std::map<std::string, AnfNodePtr>, std::map<std::string, AnfNodePtr>> local_py_params() {
    return {local_py_params_keys_, local_py_params_values_};
  }

  // Call this method to update or add a variable.
  void UpdateLocalPyParam(const std::string &name, const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    const auto key_iter = local_py_params_keys_.find(name);
    if (key_iter == local_py_params_keys_.end()) {
      MS_LOG(DEBUG) << "Add '" << name << "', " << node->DebugString();
      (void)local_py_params_keys_.emplace(std::pair<std::string, AnfNodePtr>(name, NewValueNode(name)));
      (void)local_py_params_values_.emplace(std::pair<std::string, AnfNodePtr>(name, node));
    } else {
      // Find the same position in 'values', and update the node.
      MS_LOG(DEBUG) << "Update '" << name << "', " << local_py_params_values_[name]->DebugString() << " -> "
                    << node->DebugString();
      local_py_params_values_[name] = node;
    }
  }

  // Update local parameters from previous block.
  void UpdateLocalPyParam(const std::map<std::string, AnfNodePtr> &keys, std::map<std::string, AnfNodePtr> values) {
    if (keys.size() != values.size()) {
      MS_LOG(EXCEPTION) << "keys size should be equal to values size.";
    }
    for (auto iter = keys.begin(); iter != keys.end(); ++iter) {
      const std::string &cur_key_name = iter->first;
      const auto key_iter = local_py_params_keys_.find(cur_key_name);
      if (key_iter == local_py_params_keys_.end()) {
        (void)local_py_params_keys_.emplace(std::pair<std::string, AnfNodePtr>(cur_key_name, iter->second));
        (void)local_py_params_values_.emplace(std::pair<std::string, AnfNodePtr>(cur_key_name, values[cur_key_name]));
        MS_LOG(DEBUG) << "Add '" << iter->second->DebugString() << "', " << values[cur_key_name]->DebugString();
      } else {
        // The local variable is already in the current block. This means the current block has multiples previous
        // blocks. If this local variable is used in the current block, it should be converted to phi node. So we erase
        // it from local_py_params.
        (void)local_py_params_keys_.erase(key_iter);
        (void)local_py_params_values_.erase(cur_key_name);
        MS_LOG(DEBUG) << "Erase '" << iter->second->DebugString() << "', " << values[cur_key_name]->DebugString();
      }
    }
    if (local_py_params_keys_.size() != local_py_params_values_.size()) {
      MS_LOG(EXCEPTION) << "local_py_params_keys_ size should be equal to local_py_params_values_ size.";
    }
  }

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
  std::map<std::string, std::pair<AnfNodePtr, bool>> assigned_vars_;

  // Map the parameter node to variable, it can be resolved if the block's predecessors are processed
  std::map<ParameterPtr, std::string> phi_nodes_;

  // Jumps map the successor block and the function call that perform jump
  // Refer to comments in Parser::func_block_list_ that how to break the cyclic reference
  std::map<FunctionBlock *, CNodePtr> jumps_;

  // Keep all removable phis which will be removed in one pass.
  std::map<ParameterPtr, std::set<AnfNodePtr>> phi_args_;

  // Hold declared global variables in function
  std::set<std::string> global_vars_;

  // Keep new made resolve symbol for the variable not found in vars_.
  mindspore::HashMap<std::string, AnfNodePtr> var_to_resolve_;

  // Collect all python symbols in the block.
  // We treat both global symbols and local symbols declared previously as global symbols.
  py::dict global_py_params_;
  std::map<std::string, AnfNodePtr> local_py_params_keys_;
  std::map<std::string, AnfNodePtr> local_py_params_values_;

  // Isolated nodes.
  OrderedSet<AnfNodePtr> isolated_nodes_;

  // If a block can never be executed, it's prev blocks will be empty, so this block is a dead block.
  // while x > 5:
  //    x = x - 2
  //    if x > 7 :
  //         break
  //    else :
  //         break
  //    x = x - 1   #This after block is a dead block
  bool is_dead_block_{false};

  AnfNodePtr ReadLocalVariable(const std::string &var_name);
  std::pair<AnfNodePtr, bool> FindPredInterpretNode(const std::string &var_name);
  // Flags help for determine if parallel-if transformation can be performed or not.
  // If inside this block include all inner block there is a return statement.
  // This flag will propagate beyond outer if/else or while/for loop, but not if-by-if;
  bool is_return_statement_inside_{false};
  // If inside this block there is a break/continue statement.
  // This flag will propagate beyond outer if/else but not while/for loop, if-by-if;
  bool is_break_continue_statement_inside_{false};
};
}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_FUNCTION_BLOCK_H_
