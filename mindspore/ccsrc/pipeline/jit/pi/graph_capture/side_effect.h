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
#include <string>
#include <vector>
#include <unordered_map>
#include "pipeline/jit/pi/graph_capture/node.h"

namespace mindspore {
namespace pijit {

class GlobalSideEffectNode {
 public:
  explicit GlobalSideEffectNode(std::string node_name, ValueNode *node, const char *module_name)
      : node_name_(node_name), node_(node), module_name_(module_name) {}
  virtual ~GlobalSideEffectNode() {}
  void setName(std::string node_name) { node_name_ = node_name; }
  void setNode(ValueNode *node) { node_ = node; }
  void setModule(const char *module_name) { module_name_ = module_name; }

  std::string getName() { return node_name_; }
  ValueNode *getNode() { return node_; }
  const char *getModule() { return module_name_; }

 private:
  std::string node_name_;
  ValueNode *node_;
  const char *module_name_;
};

class SideEffect {
 public:
  void setVariableMaps(ValueNode *container, ValueNode *itemNode) {
    VariableMutationMaps.insert({container, itemNode});
  }
  void setReplaceMaps(ValueNode *newNode, ValueNode *old) { replaceMaps.insert({newNode, old}); }
  int getStopBci() { return StopBci; }
  void setStopBci(int bci) { StopBci = bci; }
  std::map<ValueNode *, ValueNode *> &GetSideEffectInstrs() { return VariableMutationMaps; }
  std::map<ValueNode *, ValueNode *> &GetReplaceMaps() { return replaceMaps; }
  void ReprocessVariableMutationMaps();
  std::vector<GlobalSideEffectNode> &GetGlobalList() { return GlobalList; }

  void setGlobalList(GlobalSideEffectNode global_side_effect) { GlobalList.push_back(global_side_effect); }

 private:
  std::map<ValueNode *, ValueNode *> VariableMutationMaps;
  std::map<ValueNode *, ValueNode *> replaceMaps;
  int StopBci;
  std::vector<GlobalSideEffectNode> GlobalList;
};

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_SIDE_EFFECT_H_
