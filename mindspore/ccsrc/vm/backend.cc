/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "utils/base_ref_extends.h"
#include "backend/session/session_factory.h"
#include "utils/ms_utils.h"
#ifdef ENABLE_GE
#include "utils/callbacks_ge.h"
#endif

namespace mindspore {
namespace compile {
bool Backend::GetCond(const BaseRef &c, bool *const value) { return BaseRefToBool(c, value); }
bool Backend::GetIndex(const BaseRef &c, int *const value) { return BaseRefToInt(utils::cast<ValuePtr>(c), value); }

LinConvertResult MsBackend::MsConvert(const AnfNodePtrList &lst, const std::string &target) {
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
  GraphId graph_id = kInvalidGraphId;
  if (target != target_device_ && !target.empty()) {
    CreateOtherSession(target);
    graph_id = other_sess_->CompileGraph(lst, outputs);
  } else {
    graph_id = target_sess_->CompileGraph(lst, outputs);
  }

  if (MsContext::GetInstance()->precompile_only()) {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return result;
  }
  if (target != target_device_ && !target.empty()) {
    other_sess_->BuildGraph(graph_id);
  } else if (!is_multi_graph_sink_) {
    target_sess_->BuildGraph(graph_id);
  }
  result.run = std::make_shared<RunFunc>(
    [graph_id, target, this](const VectorRef &args) -> VectorRef { return MsRunGraph(graph_id, args, target); });
  MS_EXCEPTION_IF_NULL(result.run);

  result.simu_run = std::make_shared<RunFunc>(
    [graph_id, this](const VectorRef &args) -> VectorRef { return MsSimuRunGraph(graph_id, args); });
  MS_EXCEPTION_IF_NULL(result.simu_run);
  result.graph_id = graph_id;

  graph_id_map_[graph_id] = result;
  (void)g_ConvertCache.emplace(lst, result);
  return result;
}

// compile set input output
VectorRef MsBackend::MsSimuRunGraph(const GraphId &g, const VectorRef &args) {
  MS_LOG(DEBUG) << "set graph input:" << g;
  std::vector<BaseRef> outputs;
  (void)std::transform(graph_id_map_[g].outputs.begin(), graph_id_map_[g].outputs.end(), std::back_inserter(outputs),
                       [](const AnfNodePtr &v) { return v; });
  return VectorRef(outputs);
}

VectorRef MsBackend::MsRunGraph(const GraphId &g, const VectorRef &args, const std::string &target) {
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
    } else if (utils::isa<VectorRefPtr>(arg)) {
      auto args_new = utils::cast<VectorRef>(arg);
      (void)std::transform(args_new.begin(), args_new.end(), std::back_inserter(inputs),
                           [](const BaseRef &v) { return utils::cast<tensor::TensorPtr>(v); });
    } else {
      MS_LOG(WARNING) << "Invalid input type.";
    }
  }

  VectorRef outputs;
  // call ms rungraph (graphId, input ,output)
  if (target != target_device_ && !target.empty()) {
    other_sess_->RunGraph(g, inputs, &outputs);
  } else {
    target_sess_->RunGraph(g, inputs, &outputs);
  }

  MS_LOG(DEBUG) << "RunGraph finished:" << outputs.size();
  return outputs;
}

void MsBackend::Link(GraphId graph_id) {
  if (graph_id == kInvalidGraphId) {
    graph_id = target_sess_->GetFinalRunGraph();
  }
  target_sess_->BuildGraph(graph_id);
}

Backend::Backend(const std::string &name) : name_(name) {
  MS_LOG(DEBUG) << "select backend:" << name;
  convert_fn_ = backends[name_];
  is_multi_graph_sink_ = false;
}

MsBackend::MsBackend(const std::string &name, const std::string &target, uint32_t device_id) : Backend(name) {
  convert_fn_ = std::bind(&MsBackend::MsConvert, this, std::placeholders::_1, std::placeholders::_2);
  target_sess_ = session::SessionFactory::Get().Create(target);
  if (target_sess_ == nullptr) {
    MS_LOG(EXCEPTION) << "Session create failed!, please make sure target device:" << target << " is available.";
  }
  target_sess_->Init(device_id);
  target_sess_->RegisterSummaryCallBackFunc(callbacks::SummarySaveCallback);
  target_device_ = target;
}

void MsBackend::CreateOtherSession(const std::string &target) {
  if (other_sess_ != nullptr && other_device_ == target) {
    return;
  }
  other_sess_ = session::SessionFactory::Get().Create(target);
  if (other_sess_ == nullptr) {
    MS_LOG(EXCEPTION) << "Session create failed!, please make sure target device:" << target << " is available.";
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  uint32_t device_id = context_ptr->device_id();
  other_sess_->Init(device_id);
  other_sess_->RegisterSummaryCallBackFunc(callbacks::SummarySaveCallback);
  other_device_ = target;
}

GraphId MsBackend::CompileGraph(NotNull<FuncGraphPtr> fg) { return target_sess_->CompileGraph(fg); }

VectorRef MsBackend::RunGraph(GraphId graph_id, const VectorRef &args) { return MsRunGraph(graph_id, args); }

#ifdef ENABLE_DEBUGGER
void MsBackend::SetDebugger() { target_sess_->SetDebugger(); }
#endif

}  // namespace compile
}  // namespace mindspore
