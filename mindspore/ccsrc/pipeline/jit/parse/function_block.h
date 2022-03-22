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
  bool CollectRemovablePhi(const ParameterPtr &phi);
  // A block is matured if all its predecessors is generated
  void Mature();
  CNodePtr ForceToBoolNode(const AnfNodePtr &cond);
  CNodePtr ForceToWhileCond(const AnfNodePtr &cond);
  void Jump(const FunctionBlockPtr &block, const std::vector<AnfNodePtr> &args);
  AnfNodePtr SearchReplaceNode(const std::string &var, const ParameterPtr &phi);
  void ConditionalJump(const AnfNodePtr &cond_node, const AnfNodePtr &true_block_call,
                       const AnfNodePtr &false_block_call);
  void ConditionalJump(const AnfNodePtr &cond_node, const FunctionBlockPtr &true_block,
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
  AnfNodePtr GetResolveNode(const py::tuple &namespace_info);
  AnfNodePtr HandleNamespaceInfo(const py::tuple &namespace_info);
  AnfNodePtr HandleBuiltinNamespaceInfo(const py::tuple &namespace_info);
  AnfNodePtr MakeInterpret(const std::string &script_text, const AnfNodePtr &global_dict_node,
                           const AnfNodePtr &local_dict_node, const AnfNodePtr &orig_node);
  const mindspore::HashMap<ParameterPtr, AnfNodePtr> &removable_phis() const { return removable_phis_; }
  void FindIsolatedNodes();
  void AddIsolatedNode(const AnfNodePtr &target);
  void AttachIsolatedNodesBeforeReturn();
  const std::vector<FunctionBlock *> &prev_blocks() const { return prev_blocks_; }
  bool is_dead_block() const { return is_dead_block_; }
  void SetAsDeadBlock();

  const py::dict &global_py_params() const { return global_py_params_; }
  void set_global_py_params(const py::dict &symbols) { global_py_params_ = symbols; }
  void AddGlobalPyParam(const std::string &name, const py::object &obj) { global_py_params_[py::str(name)] = obj; }
  void UpdateGlobalPyParam(const py::dict &symbols) {
    for (auto &param : symbols) {
      if (!global_py_params_.contains(param.first)) {
        global_py_params_[param.first] = param.second;
      }
    }
  }

  std::tuple<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> local_py_params() {
    return {local_py_params_keys_, local_py_params_values_};
  }
  void AddLocalPyParam(const std::string &name, const AnfNodePtr &node) {
    MS_LOG(DEBUG) << "Add '" << name << "', " << node->DebugString();
    local_py_params_keys_.emplace_back(NewValueNode(name));
    local_py_params_values_.emplace_back(node);
  }
  // Call this methon only if you need update a variable. Usually variable override.
  void UpdateLocalPyParam(const std::string &name, const AnfNodePtr &node) {
    auto iter = std::find_if(local_py_params_keys_.cbegin(), local_py_params_keys_.cend(),
                             [&name](const AnfNodePtr node) -> bool {
                               const auto value_node = dyn_cast<ValueNode>(node);
                               MS_EXCEPTION_IF_NULL(value_node);
                               const StringImmPtr &str_imm = dyn_cast<StringImm>(value_node->value());
                               MS_EXCEPTION_IF_NULL(str_imm);
                               return name == str_imm->value();
                             });
    if (iter == local_py_params_keys_.cend()) {
      MS_LOG(EXCEPTION) << "Only for updating. Should not call this method if 'name' not exist.";
    }
    // Find the same position in 'values', and update the node.
    auto distance = std::distance(local_py_params_keys_.cbegin(), iter);
    auto values_pos_iter = local_py_params_values_.begin() + distance;
    MS_LOG(DEBUG) << "Update '" << name << "', " << (*values_pos_iter)->DebugString() << " -> " << node->DebugString();
    *values_pos_iter = node;
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
  mindspore::HashMap<ParameterPtr, AnfNodePtr> removable_phis_;

  // Hold declared global variables in function
  std::set<std::string> global_vars_;

  // Keep new made resolve symbol for the variable not found in vars_.
  mindspore::HashMap<std::string, AnfNodePtr> var_to_resolve_;

  // Collect all python symbols in the block.
  // We treat both global symbols and local symbols declared previously as global symbols.
  py::dict global_py_params_;
  std::vector<AnfNodePtr> local_py_params_keys_;
  std::vector<AnfNodePtr> local_py_params_values_;

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
};
}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_FUNCTION_BLOCK_H_
