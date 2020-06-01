/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_VM_BACKEND_H_
#define MINDSPORE_CCSRC_VM_BACKEND_H_

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "utils/contract.h"
#include "ir/anf.h"
#include "vm/segment_runner.h"
#include "vm/vm.h"
#include "session/session_basic.h"

namespace mindspore {
namespace compile {
enum SwitchCondStatus {
  kCondOk = 0,
  kCondAlreadyRun,
};

class Backend {
 public:
  explicit Backend(const std::string &name);

  virtual ~Backend() = default;

  LinkFuncType convert_fn() { return convert_fn_; }
  std::string name() { return name_; }
  virtual void SimulateRun(FinalVMPtr, FuncGraphPtr) {}
  virtual SwitchCondStatus SetSimuCond(const BaseRef &, bool) { return kCondOk; }
  virtual bool GetCond(const BaseRef &c, bool *value);
  virtual void SetSwitchGraph() {}
  virtual void SetSwitchActive(const BaseRef &, bool) {}
  virtual void RecallGraphInput(const FuncGraphPtr &, const VectorRef &, const BaseRef &) {}
  virtual void SetGraphUserInputs(const FuncGraphPtr &, const FuncGraphPtr &, const AnfNodePtrList &) {}
  virtual GraphId CompileGraph(NotNull<FuncGraphPtr> fg) { return kInvalidGraphId; }
  void set_curr_switch(const BaseRef &value) {
    curr_switch_ = value;
    is_switch_call_ = true;
  }

  BaseRef curr_switch() { return curr_switch_; }
  virtual void Link(GraphId) {}
  virtual LinConvertResult GetMultiGraphRun(const FuncGraphPtr &) { return LinConvertResult(); }

  LinConvertResult multi_result() { return multi_result_; }
  void set_multi_result(const LinConvertResult &value) { multi_result_ = value; }
  AnfNodePtr final_output() const { return final_output_; }
  bool is_multi_graph_sink() const { return is_multi_graph_sink_; }
  void set_is_multi_graph_sink(bool flag) { is_multi_graph_sink_ = flag; }
  bool simu_flag() const { return simu_flag_; }
  bool is_switch_call() const { return is_switch_call_; }
  void set_simu_flag(bool simu) { simu_flag_ = simu; }

 protected:
  std::string name_;
  LinkFuncType convert_fn_;
  BaseRef curr_switch_;  // curr switch node
  bool is_multi_graph_sink_;
  bool is_switch_call_;
  bool simu_flag_;
  LinConvertResult multi_result_;
  AnfNodePtr final_output_;
  std::unordered_map<FuncGraphPtr, std::pair<FuncGraphPtr, AnfNodePtrList>> graph_user_inputs_;
};

struct CondGraph {
  bool curr_cond;
  std::unordered_map<bool, GraphId> cond_graph_map;
};

class MsBackend : public Backend {
 public:
  MsBackend(const std::string &name, const std::string &target, uint32_t device_id);
  ~MsBackend() override = default;

  LinConvertResult MsConvert(const AnfNodePtrList &lst, const std::string &target = "");
  VectorRef MsRunGraph(const GraphId &g, const VectorRef &args, const std::string &target = "");

  VectorRef MsSimuRunGraph(const GraphId &g, const VectorRef &args);
  void SimulateRun(FinalVMPtr rt, FuncGraphPtr root) override;
  SwitchCondStatus SetSimuCond(const BaseRef &c, bool value) override;

  void SetSwitchGraph() override;
  void SetSwitchActive(const BaseRef &c, bool cond) override;
  void RecallGraphInput(const FuncGraphPtr &, const VectorRef &, const BaseRef &) override;
  void SetGraphUserInputs(const FuncGraphPtr &, const FuncGraphPtr &, const AnfNodePtrList &) override;
  void Link(GraphId) override;
  AnfNodePtr ConvertGraphInput(const FuncGraphPtr &, const AnfNodePtr &);
  LinConvertResult GetMultiGraphRun(const FuncGraphPtr &g) override;
  GraphId CompileGraph(NotNull<FuncGraphPtr> fg) override;
  VectorRef RunGraph(GraphId graph_id, const VectorRef &args);

 private:
  session::SessionPtr target_sess_;
  session::SessionPtr cpu_sess_;
  std::unordered_map<BaseRef, CondGraph, BaseRefHash> simu_cond_map_;
  std::unordered_map<GraphId, LinConvertResult> graph_id_map_;
  std::unordered_map<BaseRef, std::list<std::pair<GraphId, VectorRef>>, BaseRefHash> graph_inputs_;
};
}  // namespace compile
}  // namespace mindspore
#endif
