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
#include "pipeline/jit/pi/graph_capture/side_effect.h"
#include <algorithm>
#include <utility>
#include "pipeline/jit/pi/graph_capture/code_generator.h"
#include "pipeline/jit/pi/graph_capture/graph.h"

namespace mindspore {
namespace pijit {

ValueNode *GetSelfFromListAppendCall(ValueNode *call_node, bool *is_method_descriptor) {
  ValueNode *method_node = call_node->input(0);
  PyObject *method_object = method_node->GetVobj()->GetPyObject().ptr();
  ValueNode *self = nullptr;
  if (Py_IS_TYPE(method_object, &PyMethodDescr_Type)) {
    self = call_node->input(1);
  } else if (method_node->GetOpcode() == LOAD_ATTR) {
    self = method_node->input(0);
  }
  if (is_method_descriptor != nullptr) {
    *is_method_descriptor = Py_IS_TYPE(method_object, &PyMethodDescr_Type);
  }
  return self;
}

void SideEffectData::RecordModifiedAndReplacedNode(ValueNode *old_node, ValueNode *new_node) {
  ValueNode **old_record = &modified_and_replaced_map_[new_node];
  ValueNode *real_src = old_node;
  const auto &m = modified_and_replaced_map_;
  for (auto iter = m.find(real_src); iter != m.end(); iter = m.find(real_src)) {
    real_src = iter->second;
  }
  *old_record = real_src;
}

void SideEffectData::AddAttrData(const std::string &name, ValueNode *src, ValueNode *new_attr) {
  auto &map = attr_cache_.modified_attrs_[src];
  map[name] = new_attr;
}

void SideEffectData::AddGlobalData(const std::string &module_name, const std::string &name, ValueNode *node) {
  auto &dict = global_cache_.modified_globals_[module_name];
  dict[name] = node;
}

void SideEffectData::ClearCache() {
  attr_cache_.modified_attrs_.clear();
  global_cache_.modified_globals_.clear();
}

SideEffect::CacheResult SideEffect::LoadAttr(ValueNode *src, const std::string &name) const {
  const auto &cache = data_->attr_cache().modified_attrs_;
  if (cache.empty()) {
    return {};  // no attribute modified
  }

  CacheResult result{};
  auto Find = [&cache, &name, &result](ValueNode *src_node) {
    auto map_iter = cache.find(src_node);
    if (map_iter == cache.end()) {
      return false;  // not find attr map of this node
    }
    auto attr_iter = map_iter->second.empty() ? map_iter->second.end() : map_iter->second.find(name);
    if (attr_iter == map_iter->second.end()) {
      return false;  // not find attr of this node
    }
    result = {attr_iter->second, attr_iter->second == nullptr};
    return true;
  };

  PyObject *src_object = src->GetVobj() ? src->GetVobj()->GetPyObject().ptr() : nullptr;
  if (src_object == nullptr) {
    Find(src);
  } else {
    auto iter = data()->id_map().find(src_object);
    MS_EXCEPTION_IF_CHECK_FAIL(iter != data()->id_map().end(), "can't find the node of object");
    (void)std::find_if(iter->second.begin(), iter->second.end(), Find);
  }
  return result;
}

SideEffect::CacheResult SideEffect::LoadGlobal(const std::string &module_name, const std::string &name) const {
  const auto &cache = data_->global_cache().modified_globals_;
  if (cache.empty()) {
    return {};  // no global modified
  }
  auto m_iter = cache.find(module_name);
  if (m_iter == cache.end()) {
    return {};  // this module global not modified
  }
  auto value_iter = m_iter->second.find(name);
  if (value_iter == m_iter->second.end()) {
    return {};  // this name not modified
  }
  return {value_iter->second, value_iter->second == nullptr};
}

const std::set<ValueNode *> &SideEffect::GetRequiredNodes() const { return keep_alive_; }

bool SideEffect::Record(ValueNode *node, Type type) {
  int opcode = node->GetOpcode();
  if (opcode == STORE_ATTR || opcode == DELETE_ATTR) {
    ValueNode *src_node = opcode == DELETE_ATTR ? node->input(0) : node->input(1);
    ValueNode *attr_node = opcode == DELETE_ATTR ? nullptr : node->input(0);
    data_->AddAttrData(node->GetName(), src_node, attr_node);
    type = kSetAttr;
  } else if (opcode == STORE_GLOBAL || opcode == DELETE_GLOBAL) {
    MS_EXCEPTION_IF_NULL(node->GetGraph());
    ValueNode *new_value = opcode == DELETE_GLOBAL ? nullptr : node->input(0);
    std::string module_name = node->GetGraph()->GetModuleName();
    if (module_name.empty()) {
      return false;  // empty module name, unknown global source
    }
    data_->AddGlobalData(module_name, node->GetName(), new_value);
    type = kSetGlobal;
  } else if (opcode == STORE_SUBSCR || opcode == DELETE_SUBSCR) {
    type = kDefault;
  } else if (Opcode(opcode).IsCall() && RecordFuncCall(node, type)) {
  } else {
    MS_LOG(INFO) << "unimplemented side-effect " << node->ToString();
    return false;
  }
  size_t order_index = nodes_.size();
  nodes_[node] = {type, order_index};
  AddKeepAlive(GetKeepAlive(node, type));
  return true;
}

bool SideEffect::RecordFuncCall(ValueNode *node, Type type) {
  if (type == kDefault) {
    return true;
  }
  if (type == kSetAttr) {  // only builtin-function getattr
    size_t index = 1;
    ValueNode *src_node = node->input(index++);
    py::object name = node->input(index++)->GetVobj()->GetPyObject();
    ValueNode *attr_node = node->getInputs().size() == index ? nullptr : node->input(index);
    data_->AddAttrData(PyUnicode_AsUTF8(name.ptr()), src_node, attr_node);
    return true;
  }
  // check list.append, dict.pop, list.__setitem__, dict.__setitem__
  if (GetSelfFromListAppendCall(node) != nullptr) {
    return true;
  }
  return false;
}

std::vector<ValueNode *> SideEffect::GetKeepAlive(ValueNode *node, Type type) const {
  int opcode = node->GetOpcode();
  std::vector<ValueNode *> alive = node->getInputs();
  if (Opcode(opcode).IsCall() && type >= kListSetItem) {
    alive[0] = GetSelfFromListAppendCall(node);  // replace function
  }
  auto erase_iter = alive.begin();
  for (auto iter = erase_iter; iter != alive.end(); ++iter) {
    if (!IsNonLocalValue(*iter)) {
      *erase_iter = GetSource(*iter);
      ++erase_iter;
    }
  }
  alive.erase(erase_iter, alive.end());
  return alive;
}

void SideEffect::ResetRecord(const std::set<ValueNode *> &nodes_set) {
  // remove if record not find in final node set
  auto size = nodes_.size();
  for (auto iter = nodes_.begin(), end = nodes_.end(); iter != end;) {
    iter = nodes_set.find(iter->first) == nodes_set.end() ? nodes_.erase(iter) : (++iter);
  }
  if (size == nodes_.size()) {
    return;
  }
  // sort
  std::map<int, std::pair<ValueNode *, Type>> ordered_nodes;
  for (const auto &i : nodes_) {
    ordered_nodes[i.second.order_] = {i.first, i.second.type_};
  }
  // rollback
  keep_alive_.clear();
  nodes_.clear();
  data_->ClearCache();
  for (const auto &i : ordered_nodes) {
    this->Record(i.second.first, i.second.second);
  }
}

void SideEffect::Restore(CodeGenerator *cg) const {
  if (nodes_.empty()) {
    return;
  }
  std::vector<std::pair<ValueNode *, Type>> ordered_nodes(nodes_.size());
  for (const auto &i : nodes_) {
    ordered_nodes[i.second.order_] = {i.first, i.second.type_};
  }
  for (const auto &pair : ordered_nodes) {
    if (pair.second != SideEffect::kSetAttr && pair.second != SideEffect::kSetGlobal) {
      RestoreEntry(cg, pair.first, pair.second);
    }
  }
  RestoreAttrs(cg);
  RestoreGlobal(cg);
}

void SideEffect::RestoreEntry(CodeGenerator *cg, ValueNode *node, Type type) const {
  if (type != kDefault) {
    RestoreSpecializeEntry(cg, node, type);
    return;
  }
  int opcode = node->GetOpcode();
  int oparg = node->GetOparg();
  for (const auto &i : node->getInputs()) {
    cg->LoadValue(GetSource(i));
  }
  cg->NewInstr(opcode, oparg);
  if (Opcode(node->GetOpcode()).IsCall()) {
    cg->NewInstr(POP_TOP);
  }
}

static void MakeAttrModify(CodeGenerator *cg, const std::string &name, ValueNode *src_node, ValueNode *value) {
  auto instr = std::make_unique<Instr>(STORE_ATTR, 0, name);
  if (value != nullptr) {
    cg->LoadValue(value);
    cg->LoadValue(src_node);
  } else {
    cg->LoadValue(src_node);
    instr->set_op(DELETE_ATTR);
  }
  cg->AddInstr(std::move(instr));
}

static void MakeModuleAttrModify(CodeGenerator *cg, const std::string &name, const py::object &mod, ValueNode *value) {
  auto instr = std::make_unique<Instr>(STORE_ATTR, 0, name);
  if (value != nullptr) {
    cg->LoadValue(value);
    cg->LoadConst(mod);
  } else {
    cg->LoadConst(mod);
    instr->set_op(DELETE_ATTR);
  }
  cg->AddInstr(std::move(instr));
}

void SideEffect::RestoreAttrs(CodeGenerator *cg) const {
  if (data()->attr_cache().modified_attrs_.empty()) {
    return;
  }
  for (const auto &map : data()->attr_cache().modified_attrs_) {
    const auto &src_node = GetSource(map.first);
    for (const auto &pair : map.second) {
      MakeAttrModify(cg, pair.first, src_node, GetSource(pair.second));
    }
  }
}

void SideEffect::RestoreGlobal(CodeGenerator *cg) const {
  if (data()->global_cache().modified_globals_.empty()) {
    return;
  }
  PyObject *tmp = PyDict_GetItemString(cg->GetGlobals().ptr(), "__name__");
  const char *cur_module_name = tmp == nullptr ? "" : PyUnicode_AsUTF8(tmp);

  for (const auto &map : data()->global_cache().modified_globals_) {
    const auto &module_name = map.first;
    if (module_name != cur_module_name) {
      py::object module_object = py::reinterpret_steal<py::object>(PyImport_ImportModule(module_name.c_str()));
      for (const auto &pair : map.second) {
        MakeModuleAttrModify(cg, pair.first, module_object, GetSource(pair.second));
      }
      continue;
    }
    for (const auto &pair : map.second) {
      auto instr = std::make_unique<Instr>(STORE_GLOBAL, 0, pair.first);
      if (pair.second != nullptr) {
        cg->LoadValue(GetSource(pair.second));
      } else {
        instr->set_op(DELETE_GLOBAL);
      }
      cg->AddInstr(std::move(instr));
    }
  }
}

ValueNode *SideEffect::GetSource(ValueNode *src_node) const {
  const auto &map = data()->modified_and_replaced_map();
  if (map.empty() || src_node == nullptr) {
    return src_node;
  }
  auto iter = map.find(src_node);
  return iter != map.end() ? iter->second : src_node;
}

void SideEffect::RestoreSpecializeEntry(CodeGenerator *cg, ValueNode *node, Type type) const {
  MS_EXCEPTION_IF_CHECK_FAIL(type >= kListSetItem && type <= kDictPop, "not implemented function");
  constexpr const char *name_map[] = {"__setitem__", "__setitem__", "append", "pop"};
  static_assert(kDictPop - kListSetItem + 1 == sizeof(name_map) / sizeof(name_map[0]));
  std::string method_name = name_map[type - kListSetItem];

  bool is_method_descriptor = false;
  auto self = GetSelfFromListAppendCall(node, &is_method_descriptor);
  cg->LoadValue(GetSource(self));
  cg->AddInstr(std::make_unique<Instr>(LOAD_ATTR, 0, method_name));
  for (size_t i = 1 + is_method_descriptor; i < node->getInputs().size(); ++i) {
    cg->LoadValue(GetSource(node->input(i)));
  }
  cg->NewInstr(node->GetOpcode(), node->getInputs().size() - 1);
  cg->NewInstr(POP_TOP);
}

void SideEffect::Optimize(const std::vector<ValueNode *> &alive_locals) {
  /**
   * check data_.unique(), validate record is all in final nodes set......
   */
  // liveness analysis, remove dead local side-effect
  // not implement
  // merge dict, list modify operations
  // not implement
}

}  // namespace pijit
}  // namespace mindspore
