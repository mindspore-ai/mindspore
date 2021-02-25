/**
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
#ifndef MINDSPORE_CCSRC_DEBUG_ANF_IR_UTILS_H_
#define MINDSPORE_CCSRC_DEBUG_ANF_IR_UTILS_H_

#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/meta_func_graph.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "pipeline/jit/parse/resolve.h"
#include "frontend/operator/composite/composite.h"
#include "utils/symbolic.h"
#include "utils/ordered_map.h"
#include "utils/ordered_set.h"
#include "utils/utils.h"

namespace mindspore {

struct ParamPtrEqual {
  bool operator()(AnfNodePtr const &t1, AnfNodePtr const &t2) const {
    const ParameterPtr param1 = dyn_cast<Parameter>(t1);
    const ParameterPtr param2 = dyn_cast<Parameter>(t2);

    if (param1 == nullptr || param2 == nullptr) {
      return false;
    }

    return *param1 == *param2;
  }
};

struct ParamPtrHasher {
  std::size_t operator()(AnfNodePtr const &param) const {
    const ParameterPtr parameter = dyn_cast<Parameter>(param);
    if (parameter == nullptr) {
      return 0;
    }
    std::size_t hash = std::hash<std::string>()(parameter->name());
    return hash;
  }
};

class AnfExporter {
 public:
  explicit AnfExporter(const std::string &id, bool export_used = true, bool check_integrity = false)
      : param_index(-1), id_(id), export_used_(export_used), check_integrity_(check_integrity) {
    func_graph_set.clear();
    exported.clear();
  }
  virtual ~AnfExporter() {}

  void ExportFuncGraph(const std::string &filename, const FuncGraphPtr &func_graph);
  void ExportFuncGraph(const std::string &filename, const std::vector<TaggedGraph> &graphs);

 protected:
  virtual std::string GetNodeType(const AnfNodePtr &nd);
  int GetParamIndex(const FuncGraphPtr &func_graph, const AnfNodePtr &param, bool throw_excp = true);
  int GetParamIndexFromExported(const AnfNodePtr &param);
  std::string DumpObject(const py::object &obj, const std::string &category) const;
  std::string GetValueNodeText(const FuncGraphPtr &func_graph, const ValueNodePtr &node);
  std::string GetMultitypeFuncGraphText(const prim::MultitypeFuncGraphPtr &mt_func_graph);
  std::string GetSymbolicKeyInstanceText(const FuncGraphPtr &func_graph, const SymbolicKeyInstancePtr &sym_inst);
  std::string GetSequenceText(const FuncGraphPtr &func_graph, const ValuePtr &value);
  std::string GetValueText(const FuncGraphPtr &func_graph, const ValuePtr &value);
  std::string GetOtherValueText(const FuncGraphPtr &func_graph, const ValuePtr &value);
  std::string GetPrimitiveText(const PrimitivePtr &prim);
  std::string GetDictText(const FuncGraphPtr &func_graph, const ValuePtr &value);
  std::string GetNameSpaceText(const parse::NameSpacePtr &ns);
  std::string GetMetaFuncGraphText(const MetaFuncGraphPtr &meta_func_graph);
  std::string GetAnfNodeText(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                             const std::map<AnfNodePtr, int> &apply_map);
  virtual void ExportOneFuncGraph(std::ofstream &ofs, const FuncGraphPtr &func_graph);
  void OutputParameters(std::ofstream &ofs, const std::vector<AnfNodePtr> &parameters,
                        OrderedMap<AnfNodePtr, int, ParamPtrHasher, ParamPtrEqual> *param_map);

  void OutputStatementComment(std::ofstream &ofs, const CNodePtr &node);
  virtual void OutputCNodes(std::ofstream &ofs, const std::vector<AnfNodePtr> &nodes, const FuncGraphPtr &func_graph);
  void OutputOrderList(std::ofstream &ofs, const FuncGraphPtr &func_graph);

  int param_index;
  OrderedSet<FuncGraphPtr> func_graph_set{};
  OrderedMap<FuncGraphPtr, OrderedMap<AnfNodePtr, int, ParamPtrHasher, ParamPtrEqual>> exported;
  std::string id_;
  bool export_used_ = true;       // whether export function graphs used in current exporting function graph
  bool check_integrity_ = false;  // whether check integrity or not, when dumping ir for loading, must set it to true
  TaggedNodeMap tagged_cnodes_;
  abstract::AnfNodeConfigPtr node_cfg_ = nullptr;
};

void ExportIR(const std::string &filename, const std::string &id, const FuncGraphPtr &func_graph);
void ExportIR(const std::string &filename, const std::vector<TaggedGraph> &graphs);

std::vector<FuncGraphPtr> ImportIR(const std::string &filename);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_ANF_IR_UTILS_H_
