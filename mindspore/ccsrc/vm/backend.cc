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
#include "vm/backend.h"

#include <algorithm>
#include <vector>

#include "utils/log_adapter.h"
#include "ir/anf.h"
#include "utils/callbacks.h"
#include "utils/graph_utils.h"
#include "session/session_factory.h"
#include "common/utils.h"
#ifdef ENABLE_GE
#include "utils/callbacks_ge.h"
#endif

namespace mindspore {
namespace compile {
bool Backend::GetCond(const BaseRef &c, bool *const value) { return BaseRefToBool(c, value); }

LinConvertResult MsBackend::GetMultiGraphRun(const FuncGraphPtr &g) {
  // multi_graph merge to one, big graph have paramters in begin and only have one output
  MS_LOG(DEBUG) << "graph:" << g->ToString() << " parameter size:" << g->parameters().size();
  multi_result_.inputs = g->parameters();
  final_output_ = NewValueNode("fake_output");
  multi_result_.outputs = {final_output_};
  GraphId final_g = sess_->GetFinalRunGraph();

  multi_result_.run = std::make_shared<RunFunc>(
    [final_g, this](const VectorRef &args) -> VectorRef { return MsRunGraph(final_g, args); });
  return multi_result_;
}

LinConvertResult MsBackend::MsConvert(const AnfNodePtrList &lst) {
  MS_LOG(DEBUG) << "MsConvert";
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  auto cached = g_ConvertCache.find(lst);
  if (cached != g_ConvertCache.end()) {
    return cached->second;
  }

  LinConvertResult result;

  FuncGraphPtr fg;
  AnfNodePtrList inputs;
  AnfNodePtrList outputs;

  std::tie(fg, inputs, outputs) = TransformSegmentToAnfGraph(lst);
  result.inputs = inputs;
  result.outputs = outputs;
  result.graph_id = kInvalidGraphId;
  auto graph_id = sess_->CompileGraph(lst, outputs);
  if (MsContext::GetInstance()->precompile_only()) {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return result;
  }

  result.run = std::make_shared<RunFunc>(
    [graph_id, this](const VectorRef &args) -> VectorRef { return MsRunGraph(graph_id, args); });
  MS_EXCEPTION_IF_NULL(result.run);

  result.simu_run = std::make_shared<RunFunc>(
    [graph_id, this](const VectorRef &args) -> VectorRef { return MsSimuRunGraph(graph_id, args); });
  MS_EXCEPTION_IF_NULL(result.simu_run);
  result.graph_id = graph_id;

  graph_id_map_[graph_id] = result;
  (void)g_ConvertCache.emplace(lst, result);
  return result;
}

void MsBackend::SetSwitchActive(const BaseRef &c, bool cond) {
  GraphId active_g = simu_cond_map_[c].cond_graph_map[cond];

  GraphId cond_g = kInvalidGraphId;
  if (utils::isa<AnfNodePtr>(c)) {
    cond_g = sess_->GetGraphIdByNode(utils::cast<AnfNodePtr>(c));
  } else {
    MS_LOG(EXCEPTION) << "cond not a anf node:" << c.ToString();
  }
  auto before_cond = curr_switch_;
  if (curr_switch_.hash() != c.hash()) {
    // invoke while false->before true call
    if (simu_cond_map_[before_cond].cond_graph_map.count(false)) {
      active_g = simu_cond_map_[before_cond].cond_graph_map[false];
    } else {
      active_g = kInvalidGraphId;
    }
    // while x < y:
    //   z = y + 1
    //   while z < c2:
    //     out = out + 1
    //     z = z + 1
    if (active_g == cond_g) {
      active_g = kInvalidGraphId;
      simu_cond_map_[before_cond].cond_graph_map[false] = kInvalidGraphId;
    }
    MS_LOG(DEBUG) << "invoke set active:" << active_g;
  }
  MS_LOG(DEBUG) << "switch set active:" << active_g << ", " << cond_g;
  sess_->SetActive(active_g, cond_g);
}

void MsBackend::SetSwitchGraph() {
  MS_LOG(DEBUG) << "SetSwitchGraph curr_switch:" << curr_switch_.ToString();

  if (is_switch_call_) {
    GraphId false_g = kInvalidGraphId;
    GraphId true_g = kInvalidGraphId;
    MS_LOG(DEBUG) << "start SetSwitchGraph";
    true_g = simu_cond_map_[curr_switch_].cond_graph_map[true];
    bool curr_cond = simu_cond_map_[curr_switch_].curr_cond;
    if (!curr_cond) {
      if (simu_cond_map_[curr_switch_].cond_graph_map.count(curr_cond)) {
        // has false branch
        false_g = simu_cond_map_[curr_switch_].cond_graph_map[false];
      }
      GraphId cond_g = kInvalidGraphId;
      if (utils::isa<AnfNodePtr>(curr_switch_)) {
        cond_g = sess_->GetGraphIdByNode(utils::cast<AnfNodePtr>(curr_switch_));
      } else {
        MS_LOG(EXCEPTION) << "cond not a anf node:" << curr_switch_.ToString();
      }
      MS_LOG(DEBUG) << "switch compile:" << cond_g << ", " << true_g << ", " << false_g;
      sess_->SwitchCompile(cond_g, true_g, false_g);
    }
    is_switch_call_ = false;
    MS_LOG(DEBUG) << "end SetSwitchGraph:" << curr_cond << ", " << is_switch_call_;
  }
}

// compile set input output
VectorRef MsBackend::MsSimuRunGraph(const GraphId &g, const VectorRef &args) {
  MS_LOG(DEBUG) << "set graph input:" << g;
  // switch maybe twice
  sess_->SetChildGraphInput(g, args);

  if (is_switch_call_) {
    bool curr_cond = simu_cond_map_[curr_switch_].curr_cond;
    MS_LOG(DEBUG) << "switch call MsSimuRunGraph:" << curr_cond;
    if (0 == simu_cond_map_[curr_switch_].cond_graph_map.count(curr_cond)) {
      MS_LOG(DEBUG) << "switch call MsSimuRunGraph:" << curr_cond << ", " << g;
      simu_cond_map_[curr_switch_].cond_graph_map[curr_cond] = g;
      SetSwitchGraph();
    }
  }

  std::vector<BaseRef> outputs;
  (void)std::transform(graph_id_map_[g].outputs.begin(), graph_id_map_[g].outputs.end(), std::back_inserter(outputs),
                       [](const AnfNodePtr &v) { return v; });
  return VectorRef(outputs);
}

VectorRef MsBackend::MsRunGraph(const GraphId &g, const VectorRef &args) {
  MS_LOG(DEBUG) << "start ms graph run:" << args.size() << ", g:" << g;
  // Run graph
  std::vector<tensor::TensorPtr> inputs;
  for (const auto &arg : args) {
    if (utils::isa<tensor::TensorPtr>(arg)) {
      auto value = utils::cast<tensor::TensorPtr>(arg);
      inputs.push_back(value);
    } else if (utils::isa<ValuePtr>(arg)) {
      auto value = utils::cast<ValuePtr>(arg);
      if (value->isa<ValueTuple>()) {
        (void)std::transform(value->cast<ValueTuplePtr>()->value().begin(), value->cast<ValueTuplePtr>()->value().end(),
                             std::back_inserter(inputs),
                             [](const ValuePtr &v) { return v->cast<tensor::TensorPtr>(); });
      } else if (value->isa<Scalar>()) {
        tensor::TensorPtr scalar_tensor = ScalarToTensor(value->cast<ScalarPtr>());
        MS_EXCEPTION_IF_NULL(scalar_tensor);
        inputs.push_back(scalar_tensor);
      } else {
        inputs.push_back(value->cast<tensor::TensorPtr>());
      }
    } else if (utils::isa<PyObjectRef>(arg)) {
      auto value = utils::cast<PyObjectRef>(arg).object_;
      inputs.push_back(py::cast<tensor::TensorPtr>(value));
    }
  }

  VectorRef outputs;
  // call ms rungraph (graphId, input ,output)
  sess_->RunGraph(g, inputs, &outputs);
  MS_LOG(DEBUG) << "RunGraph finished:" << outputs.size();
  return outputs;
}

void MsBackend::SetSimuCondFlag(const BaseRef &c, int flag) {
  MS_LOG(DEBUG) << "while set cond :" << c.ToString() << ", " << simu_cond_map_.size();

  if (simu_cond_map_.find(c) == simu_cond_map_.end()) {
    MS_LOG(EXCEPTION) << "error c not find";
  }
  simu_cond_map_[c].flag = flag;
}

int MsBackend::GetSimuCondFlag(const BaseRef &c) {
  BaseRef cond = c;
  if (cond.is_null()) {
    MS_LOG(DEBUG) << "get curr_switch";
    cond = curr_switch_;
  }
  if (simu_cond_map_.find(cond) == simu_cond_map_.end()) {
    MS_LOG(ERROR) << "error c not find";
    return -1;
  }
  return simu_cond_map_[cond].flag;
}

SwitchCondStatus MsBackend::SetSimuCond(const BaseRef &c, bool value) {
  MS_LOG(DEBUG) << "set cond :" << c.ToString() << ", " << simu_cond_map_.size();

  CondGraph cond_graph;
  cond_graph.curr_cond = value;
  if (simu_cond_map_.find(c) == simu_cond_map_.end()) {
    cond_graph.flag = 0;
    simu_cond_map_[c] = cond_graph;
  }

  if (simu_cond_map_[c].cond_graph_map.count(value)) {
    if (value == true) {
      return kCondAlreadyRun;
    }
  }
  simu_cond_map_[c].curr_cond = value;
  MS_LOG(DEBUG) << "end set cond ";
  return kCondOk;
}

void MsBackend::SimulateRun(FinalVMPtr rt, FuncGraphPtr root) {
  MS_LOG(DEBUG) << "Simulate run,root:" << root->ToString() << ", " << root->parameters().size();
  std::vector<BaseRef> args;
  auto parameters = root->parameters();
  (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args),
                       [](const AnfNodePtr &v) { return v; });
  MS_LOG(DEBUG) << "Simulate start";
  (void)sess_->SetFinalGraphInput(parameters);
  BaseRef output = rt->Eval(VectorRef(args));
  sess_->SetFinalGraphOutput(output);
  MS_LOG(DEBUG) << "Simulate Eval end";
}

void MsBackend::Link(GraphId graph_id) {
  if (graph_id == kInvalidGraphId) {
    graph_id = sess_->GetFinalRunGraph();
  }
  sess_->BuildGraph(graph_id);
}

Backend::Backend(const std::string &name) : name_(name) {
  MS_LOG(DEBUG) << "select backend:" << name;
  convert_fn_ = backends[name_];
  is_switch_call_ = false;
  is_multi_graph_sink_ = false;
  simu_flag_ = false;
}

MsBackend::MsBackend(const std::string &name, const std::string &target, uint32_t device_id) : Backend(name) {
  convert_fn_ = std::bind(&MsBackend::MsConvert, this, std::placeholders::_1);
  sess_ = session::SessionFactory::Get().Create(target);
  if (sess_ == nullptr) {
    MS_LOG(EXCEPTION) << "Session create failed!, please make sure target device:" << target << " is available.";
  }
  sess_->Init(device_id);
  sess_->RegisterSummaryCallBackFunc(callbacks::SummarySaveCallback);
}

}  // namespace compile
}  // namespace mindspore
