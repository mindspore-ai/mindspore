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

#include <memory>
#include <vector>
#include <map>
#include <set>
#include <string>
#include "pipeline/jit/pi/graph_capture/node.h"

namespace mindspore {
namespace pijit {

class CodeGenerator;

// an unique data in the whole compilation
class SideEffectData {
 public:
  struct AttrCache {
    // a map of the modified object and it's modified attrs
    using AttrMap = std::map<std::string, ValueNode *>;
    std::map<ValueNode *, AttrMap> modified_attrs_;
  };

  struct GlobalCache {
    // a map of module and modified global dict
    using NameMap = std::map<std::string, ValueNode *>;
    std::map<std::string, NameMap> modified_globals_;
  };

  const auto &attr_cache() const { return attr_cache_; }
  const auto &global_cache() const { return global_cache_; }
  const auto &id_map() const { return id_map_; }
  const auto &modified_and_replaced_map() const { return modified_and_replaced_map_; }

  // track object and nodes
  void Track(PyObject *ptr, ValueNode *node) { (ptr ? (void)id_map_[ptr].insert(node) : (void)0); }
  void UnTrack(PyObject *ptr, ValueNode *node) { (ptr ? (void)id_map_[ptr].erase(node) : (void)0); }

  // record replaced node
  void RecordModifiedAndReplacedNode(ValueNode *src_node, ValueNode *new_node);

  // merge attr modify operations
  void AddAttrData(const std::string &name, ValueNode *src, ValueNode *new_attr);

  // merge global modify operations
  void AddGlobalData(const std::string &module_name, const std::string &name, ValueNode *value);

  void ClearCache();

 private:
  // an unique map that record python object and nodes in the whole compilation
  // used to resolve object consistency
  std::map<PyObject *, std::set<ValueNode *>> id_map_;

  // an unique map of new value(key) and old_value(value)
  std::map<ValueNode *, ValueNode *> modified_and_replaced_map_;

  // optimization cache, record modified object
  // if record is reset, clean cache
  AttrCache attr_cache_;
  GlobalCache global_cache_;
};

class SideEffect {
 public:
  enum Type {
    kDefault,
    kSetAttr,
    kSetGlobal,
    kListSetItem,
    kDictSetItem,
    kListAppend,
    kDictPop,
  };

  struct CacheResult {
    ValueNode *cache_value_;
    bool is_deleted_value_;
  };

  // find attribute from id_map and attr cache
  CacheResult LoadAttr(ValueNode *src, const std::string &name) const;

  // find global from global cache
  CacheResult LoadGlobal(const std::string &module_name, const std::string &name) const;

 public:
  SideEffect() = default;

  const auto &data() const { return data_; }
  void set_data(const std::shared_ptr<SideEffectData> &data) { data_ = data; }

  // check the node is a side-effect record
  bool IsRecord(ValueNode *node) const { return nodes_.empty() ? false : nodes_.find(node) != nodes_.end(); }

  // check record is empty
  bool IsEmpty() const { return nodes_.empty(); }

  // return false if unsupported the side-effect
  bool Record(ValueNode *side_effect_node, Type type = Type::kDefault);

  // generate the code to restore side-effect
  void Restore(CodeGenerator *cg) const;

  // reset the record if record not find in final nodes set
  void ResetRecord(const std::set<ValueNode *> &traced_nodes);

  // return the original node(source) if it's replaced, else return the node
  ValueNode *GetSource(ValueNode *node) const;

  // optimize the side-effect data, remove modify operations of dead local variable
  void Optimize(const std::vector<ValueNode *> &alive_locals);

  // return the side-effect handler required nodes
  const std::set<ValueNode *> &GetRequiredNodes() const;

 private:
  // add nodes to required
  void AddKeepAlive(const std::vector<ValueNode *> &inputs) { keep_alive_.insert(inputs.begin(), inputs.end()); }

  // get required node of the side-effect node
  std::vector<ValueNode *> GetKeepAlive(ValueNode *node, Type type) const;

  // if side-effect is function call, check it's supported
  bool RecordFuncCall(ValueNode *node, Type type);

  // restore a side-effect node
  void RestoreEntry(CodeGenerator *cg, ValueNode *node, Type type) const;

  // restore attribute
  void RestoreAttrs(CodeGenerator *cg) const;

  // restore global
  void RestoreGlobal(CodeGenerator *cg) const;

  // restore list, dict, or other specialized object function call
  void RestoreSpecializeEntry(CodeGenerator *cg, ValueNode *node, Type type) const;

  struct Entry {
    Type type_;
    size_t order_;
  };

  // shared from other side-effect recorder
  std::shared_ptr<SideEffectData> data_;

  // record operations, check side-effect order
  std::map<ValueNode *, Entry> nodes_;

  // side-effect handler required nodes
  std::set<ValueNode *> keep_alive_;
};

// return the self node, if return nullptr, unsupported to handle side-effect
ValueNode *GetSelfFromListAppendCall(ValueNode *call_node, bool *is_method_descriptor = nullptr);

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_SIDE_EFFECT_H_
