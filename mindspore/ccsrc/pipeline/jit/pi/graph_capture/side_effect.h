/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_SIDE_EFFECT_H_
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_SIDE_EFFECT_H_

#include <set>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "pipeline/jit/pi/graph_capture/node.h"

namespace mindspore {
namespace pijit {

class CodeGenerator;

class GlobalSideEffectNode {
 public:
  explicit GlobalSideEffectNode(std::string node_name, ValueNode *node, const char *module_name)
      : node_name_(node_name), node_(node), module_name_(module_name) {}
  virtual ~GlobalSideEffectNode() {}
  void setName(std::string node_name) { node_name_ = node_name; }
  void setNode(ValueNode *node) { node_ = node; }
  void setModule(const char *module_name) { module_name_ = module_name; }

  std::string getName() const { return node_name_; }
  ValueNode *getNode() const { return node_; }
  const char *getModule() { return module_name_; }

 private:
  std::string node_name_;
  ValueNode *node_;
  const char *module_name_;
};

class SideEffect {
 public:
  void SetSideEffectNode(ValueNode *node) { side_effect_nodes.push_back(node); }
  std::vector<ValueNode *> &GetSideEffectNodes() { return side_effect_nodes; }
  std::vector<ValueNode *> const &GetSideEffectNodes() const { return side_effect_nodes; }

  void SetReplaceMap(ValueNode *newNode, ValueNode *oldNode) { replace_map.insert({newNode, oldNode}); }
  std::map<ValueNode *, ValueNode *> &GetReplaceMap() { return replace_map; }
  std::map<ValueNode *, ValueNode *> const &GetReplaceMap() const { return replace_map; }

  void ConvertReplaceList();
  void SetReplaceList(ValueNode *node) { replace_list.push_back(node); }
  std::vector<ValueNode *> &GetReplaceList() { return replace_list; }
  std::vector<ValueNode *> const &GetReplaceList() const { return replace_list; }

  void SetGlobalList(GlobalSideEffectNode global_side_effect) { global_list.push_back(global_side_effect); }
  std::vector<GlobalSideEffectNode> &GetGlobalList() { return global_list; }
  std::vector<GlobalSideEffectNode> const &GetGlobalList() const { return global_list; }

  std::vector<ValueNode *> CollectSideEffectAliveNodes() const;
  void CleanSideEffects(int new_bci);
  void RestoreSideEffect(CodeGenerator *) const;
  void Merge(const std::unique_ptr<SideEffect> &sub_side_effect);

 private:
  std::vector<ValueNode *> side_effect_nodes;
  std::map<ValueNode *, ValueNode *> replace_map;
  std::vector<ValueNode *> replace_list;
  std::vector<GlobalSideEffectNode> global_list;
};

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_SIDE_EFFECT_H_
