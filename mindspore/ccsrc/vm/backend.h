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
#include "vm/graph_partition.h"
#include "vm/vm.h"
#include "backend/session/session_basic.h"

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
  virtual bool GetCond(const BaseRef &c, bool *value);
  virtual bool GetIndex(const BaseRef &c, int64_t *value);
  virtual GraphId CompileGraph(NotNull<FuncGraphPtr> fg) { return kInvalidGraphId; }
  virtual void Link(GraphId) {}
  virtual void SetDebugger() {}

  bool is_multi_graph_sink() const { return is_multi_graph_sink_; }
  void set_is_multi_graph_sink(bool flag) { is_multi_graph_sink_ = flag; }

 protected:
  std::string name_;
  LinkFuncType convert_fn_;
  bool is_multi_graph_sink_;
};

class MsBackend : public Backend {
 public:
  MsBackend(const std::string &name, const std::string &target, uint32_t device_id);
  ~MsBackend() override = default;

  LinConvertResult MsConvert(const GraphSegmentPtr &segment, const std::string &target = "");
  VectorRef MsRunGraph(const GraphId &g, const VectorRef &args, const std::string &target = "");

  VectorRef MsSimuRunGraph(const GraphId &g, const VectorRef &args);
  void Link(GraphId) override;
  GraphId CompileGraph(NotNull<FuncGraphPtr> fg) override;
  VectorRef RunGraph(GraphId graph_id, const VectorRef &args);
  void ClearSessionGraphs();
  void CreateOtherSession(const std::string &target);

#ifdef ENABLE_DEBUGGER
  void SetDebugger() override;
#endif

 private:
  session::SessionPtr target_sess_;
  session::SessionPtr other_sess_;
  std::string target_device_;
  std::string other_device_;
  std::unordered_map<GraphId, LinConvertResult> graph_id_map_;
};
}  // namespace compile
}  // namespace mindspore
#endif
