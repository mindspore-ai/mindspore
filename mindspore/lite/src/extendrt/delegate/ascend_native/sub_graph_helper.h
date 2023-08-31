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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_SUB_GRAPH_HELPER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_SUB_GRAPH_HELPER_H_

#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include <string>
#include "ir/anf.h"
#include "mindapi/base/type_id.h"
#include "extendrt/delegate/ops/copy.h"

namespace mindspore {
class SubGraphHelper;
using SubGraphHelperPtr = std::shared_ptr<SubGraphHelper>;

class AnSubGraph {
 public:
  AnSubGraph() = delete;
  explicit AnSubGraph(int index);
  void Add(const CNodePtr &cnode);
  int Size() const;
  void AddInput(const AnfNodePtr &node);
  void AddOutput(const AnfNodePtr &node);
  const std::vector<AnfNodePtr> &inputs() const { return inputs_; }
  void set_cnode(const CNodePtr &cnode) { cnode_ = cnode; }
  const CNodePtr &cnode() const { return cnode_; }
  int GetOutputId(const CNodePtr &cnode) const;
  int GetOutputsCount() const { return outputs_.size(); }
  void FixGroup(SubGraphHelperPtr helper);
  void Dump();
  const FuncGraphPtr &func_graph() { return func_graph_; }
  void SetAbstract();

 private:
  int index_{0};
  CNodePtr CreateTuple();
  void DumpNode(const AnfNodePtr &node);
  FuncGraphPtr func_graph_;
  CNodePtr cnode_;
  std::vector<AnfNodePtr> inputs_;
  std::unordered_set<AnfNodePtr> input_set_;
  std::vector<AnfNodePtr> outputs_;
  std::unordered_set<AnfNodePtr> output_set_;
};

class SubGraphHelper : public std::enable_shared_from_this<SubGraphHelper> {
  using SubGraphPtr = std::shared_ptr<AnSubGraph>;

 public:
  SubGraphHelper() = delete;
  explicit SubGraphHelper(const FuncGraphPtr &graph) : func_graph_{graph} {}
  int CheckAllInputInSameSg(const CNodePtr &cnode);
  int FindSubGraph(const AnfNodePtr &node) const;
  void AddSubGraph(const CNodePtr &node);
  void AddToSubGraph(int index, const CNodePtr &node, bool update = true);
  int SubGroupNum() const { return sg_v_.size(); }
  const std::vector<AnfNodePtr> &GetSbgInputs(int idx) { return GetSbg(idx)->inputs(); }
  SubGraphPtr &CreateSubGraph();
  const CNodePtr &GetCNode(int idx) const;
  void DrawGraph(const std::string &file_name, const FuncGraphPtr &graph, bool recursive = false) const;
  void SetCNode(int idx, const CNodePtr &cnode);
  void FixAllNodes(const AnfNodePtrList &nodes);
  const SubGraphPtr &GetSbg(int i) const { return sg_v_[i]; }
  AnfNodePtr CreateGetItemAndCopyUnique(const AnfNodePtr &node, int id, const CNodePtr &cinput,
                                        ops::Copy::CopyFormatType type);
  bool IsGraphInput(const AnfNodePtr &node) const;
  void Dump(std::string file_name) const;

 private:
  int GetOutputsCount(int group);
  int GetOutputId(int group, const CNodePtr &input) const;
  void AddSubGraphOutput(int group, const CNodePtr &cnode) { GetSbg(group)->AddOutput(cnode); }
  CNodePtr CreateGetItem(const AnfNodePtr &node, int id, const CNodePtr &input);
  CNodePtr CreateCopyNode(const AnfNodePtr &input, ops::Copy::CopyFormatType type);
  void SetOutputsAndAbstract(const AnfNodePtrList &nodes);
  void UpdateInput(const CNodePtr &cnode, int index, const AnfNodePtr &input) const;
  void FixOutput();
  void FixGroups();
  void DrawGraph(const FuncGraphPtr &graph, std::ostream &out, bool recursive) const;
  void DrawConnction(const AnfNodePtr &in_node, bool src_composite, int src_idx, const AnfNodePtr &node,
                     bool dst_composite, int dst_idx, std::ostream &out) const;
  void DumpNode(std::ofstream &out, const AnfNodePtr &node) const;
  std::vector<SubGraphPtr> sg_v_;
  std::unordered_map<CNodePtr, int> map_;
  std::map<std::pair<int, AnfNodePtr>, AnfNodePtr> connection_map_;  // <port, node>->node
  const FuncGraphPtr &func_graph_;
};
};      // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_SUB_GRAPH_HELPER_H_
