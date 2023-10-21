/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_H_
#define MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_H_

#include <map>
#include <set>
#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <unordered_map>
#include "ir/anf.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"

namespace mindspore {
namespace expander {
namespace bprop {
bool HasBpropExpander(const std::string &prim_name);
void ClearBpropOpGraphMap();
using UserType = std::unordered_map<AnfNodePtr, std::vector<std::pair<std::weak_ptr<CNode>, int>>>;
struct UserMap {
  UserType dout_user_;
  UserType tuple_getitem_user_;
};
class BpropExpander {
 public:
  BpropExpander(CNodePtrList *outputs, UserMap *users) : outputs_(outputs), users_(users) {}
  ~BpropExpander() = default;
  bool Run(const CNodePtr &cnode, const std::vector<ValuePtr> &input_values = {});
  static const mindspore::HashSet<size_t> &GetUnusedInputs(const string &op_name);

 protected:
  bool RunBprop(const CNodePtr &cnode, const std::vector<ValuePtr> &input_values);
  void PostProcess(const CNodePtr &cnode) const;
  void DumpResult(const std::string &name) const;
  NodePtrList input_nodes_;
  NodePtrList output_nodes_;
  CNodePtrList *outputs_{nullptr};
  UserMap *users_{nullptr};
};

bool ExpandBpropInGraphMode(const BpropHandle *handle, const PrimitivePtr &prim, const FuncGraphPtr &graph);

class OpEnvManager {
 public:
  static bool UsePyBprop(const std::string &name) {
    static const auto op_set = GetEnvSet();
    return op_set.count(name) != 0;
  }

 private:
  static std::set<std::string> GetEnvSet() {
    auto env = common::GetEnv("MS_DEV_USE_PY_BPROP");
    if (env.empty()) {
      return {};
    }
    std::set<std::string> op_set;
    std::stringstream ss(env);
    std::string token;
    std::ostringstream oss;
    while (std::getline(ss, token, ',')) {
      if (op_set.insert(token).second) {
        oss << "\"" << token << "\",";
      }
    }
    MS_LOG(INFO) << "Env \"MS_DEV_USE_PY_BPROP\" set ops: " << oss.str();
    return op_set;
  }
};
#ifdef _MSC_VER
class WinBpropRegister {
 public:
  WinBpropRegister();
  ~WinBpropRegister() {}
  void DoNothing() const {}
};
#endif
}  // namespace bprop
}  // namespace expander

using expander::bprop::BpropExpander;
#ifdef _MSC_VER
using expander::bprop::WinBpropRegister;
#endif
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_H_
