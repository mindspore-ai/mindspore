/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "backend/ge_backend/graph_ir/utils.h"
#include "backend/ge_backend/graph_ir/aoe_util.h"
#include "backend/ge_backend/graph_ir/convert.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_map.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_util.h"
#include "backend/ge_backend/graph_ir/df_graph_manager.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_desc.h"
#include "plugin/ascend/res_manager/op_adapter/transform_util.h"
#include "backend/ge_backend/graph_ir/graph_builder.h"
#include "include/utils/anfalgo.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore::backend::ge_backend {

void ClearGeSessionAndRunner() {
  DfGraphManager::GetInstance().DeleteGraphRunner();
  DfGraphManager::GetInstance().DeleteGeSession();
  DfGraphManager::GetInstance().ClearGraph();
}

bool IsInitDataSetQueueNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsPrimitiveCNode(cnode, prim::kPrimInitDataSetQueue)) {
    return true;
  }
  return false;
}

std::vector<GeTensorPtr> ConvertInputTensors(const std::vector<MeTensorPtr> &me_tensors, const std::string &format) {
  return device::ascend::TransformUtil::ConvertInputTensors(me_tensors, format);
}

std::vector<MeTensorPtr> ConvertGeTensors(const std::vector<GeTensorPtr> &ge_tensors) {
  return device::ascend::TransformUtil::ConvertGeTensors(ge_tensors);
}

GeDataType ConvertDataType(const MeDataType &type) { return device::ascend::TransformUtil::ConvertDataType(type); }

MeTensorPtr ConvertGeTensor(const GeTensorPtr &ge_tensor, const ShapeVector &request_dims, bool ref_mem) {
  return device::ascend::TransformUtil::ConvertGeTensor(ge_tensor, request_dims, ref_mem);
}

MeTensorPtr ConvertGeTensor(const GeTensorPtr &tensor) {
  return device::ascend::TransformUtil::ConvertGeTensor(tensor);
}

MeTensorPtr ConvertGeTensor(const GeTensorPtr &tensor, const TypeId &me_type) {
  return device::ascend::TransformUtil::ConvertGeTensor(tensor, me_type);
}

std::shared_ptr<backend::ge_backend::GraphRunner> GetGraphRunner() {
  return DfGraphManager::GetInstance().GetGraphRunner();
}

std::shared_ptr<backend::ge_backend::GraphRunner> CheckAndGetGraphRunner(
  const backend::ge_backend::RunOptions &run_options) {
  if (backend::ge_backend::GetGraphByName(run_options.name) == nullptr) {
    MS_LOG(WARNING) << "Can not find " << run_options.name
                    << " sub graph, don't need data init subgraph in INFER mode.";
    return nullptr;
  }

  auto graph_runner = backend::ge_backend::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }
  return graph_runner;
}

std::shared_ptr<::ge::Session> GetGeSession() { return DfGraphManager::GetInstance().GetGeSession(); }

void SetGeSession(const std::shared_ptr<::ge::Session> &sess_ptr) {
  DfGraphManager::GetInstance().SetGeSession(sess_ptr);
}

GraphRunnerPtr NewGraphRunner(const GraphRunnerOptions &options) {
  auto graph_runner = std::make_shared<backend::ge_backend::GraphRunner>(options);
  return graph_runner;
}

void SetGraphRunner(const GraphRunnerPtr &runner) { DfGraphManager::GetInstance().SetGraphRunner(runner); }
void ClearGraph() { DfGraphManager::GetInstance().ClearGraph(); }

Status AddGraph(const std::string &name, const DfGraphPtr &graph, const DfGraphConfig &graph_config) {
  auto ret = DfGraphManager::GetInstance().AddGraph(name, graph, graph_config);
  if (ret != Status::SUCCESS) {
    return ret;
  }
  if (graph_config.need_aoe_) {
    backend::ge_backend::AddOptimizeGraph(name);
    backend::ge_backend::DfGraphManager::GetInstance().AoeGeGraph();
  }
  auto graph_runner = backend::ge_backend::GetGraphRunner();
  if (graph_runner == nullptr) {
    // lite may not use graph_runner
    MS_LOG(INFO) << "There is no GraphRunner.";
    return ret;
  }
  return graph_runner->AddGraph(name);
}

void SetAnfGraph(const std::string &name, const AnfGraphPtr &anf_graph_ptr) {
  DfGraphManager::GetInstance().SetAnfGraph(name, anf_graph_ptr);
}

FuncGraphPtr GetAnfGraph(uint32_t graph_id) { return DfGraphManager::GetInstance().GetAnfGraph(graph_id); }

DfGraphWrapperPtr GetGraphByName(const std::string &name) { return DfGraphManager::GetInstance().GetGraphByName(name); }

void AddOptimizeGraph(const std::string &name) { AoeUtil::GetInstance().AddOptimizeGraph(name); }

void InitializeAoeUtil(const std::string &aoe_job_type) { AoeUtil::GetInstance().Initialize(aoe_job_type); }

void DestroyAoeUtil() { AoeUtil::GetInstance().Destroy(); }

void EnableAoeOffline() { AoeUtil::GetInstance().SetOfflineEnvDumpGeGraph(); }

// convert

DfGraphConvertorPtr NewConverter(const FuncGraphPtr &graph, const std::string &phase_prefix, RefModeFlag ref_mode_type,
                                 bool offline_convert) {
  std::vector<std::string> extra_variables_names = {};
  auto converter = std::make_shared<backend::ge_backend::DfGraphConvertor>(
    graph, phase_prefix, ref_mode_type, extra_variables_names, nullptr, offline_convert);
  return converter;
}

void SetTraining(const DfGraphConvertorPtr &converter, bool training) {
  MS_EXCEPTION_IF_NULL(converter);
  converter->set_training(training);
}

void SetExportAir(const DfGraphConvertorPtr &converter, bool export_air) {
  MS_EXCEPTION_IF_NULL(converter);
  converter->set_export_air(export_air);
}

void BuildGraph(const std::string &name, const DfGraphConvertorPtr &converter,
                const std::map<std::string, std::shared_ptr<tensor::Tensor>> &maps) {
  MS_EXCEPTION_IF_NULL(converter);
  (void)converter->ConvertAllNode().InitParam(maps).BuildGraph(name);
}

void GenerateBroadcastGraph(const DfGraphConvertorPtr &converter, const TensorOrderMap &tensors) {
  MS_EXCEPTION_IF_NULL(converter);
  (void)converter->GenerateBroadcastGraph(tensors);
}
void GenerateCheckpointGraph(const DfGraphConvertorPtr &converter) {
  MS_EXCEPTION_IF_NULL(converter);
  (void)converter->GenerateCheckpointGraph();
}
int ErrCode(const DfGraphConvertorPtr &converter) {
  MS_EXCEPTION_IF_NULL(converter);
  return converter->ErrCode();
}

void GenFakeGraph(const std::string &name, const DfGraphConvertorPtr &converter) {
  MS_EXCEPTION_IF_NULL(converter);
  converter->GenFakeGraph(name);
}

DfGraphPtr GetComputeGraph(const DfGraphConvertorPtr &converter) {
  MS_EXCEPTION_IF_NULL(converter);
  return converter->GetComputeGraph();
}
DfGraphPtr GetInitGraph(const DfGraphConvertorPtr &converter) {
  MS_EXCEPTION_IF_NULL(converter);
  return converter->GetInitGraph();
}
DfGraphPtr GetSaveCheckpointGraph(const DfGraphConvertorPtr &converter) {
  MS_EXCEPTION_IF_NULL(converter);
  return converter->GetSaveCheckpointGraph();
}
DfGraphPtr GetBroadcastGraph(const DfGraphConvertorPtr &converter) {
  MS_EXCEPTION_IF_NULL(converter);
  return converter->GetBroadcastGraph();
}

std::shared_ptr<::ge::Session> NewSession(const SessionOptions &sess_options) {
  return backend::ge_backend::GraphRunner::NewSession(sess_options);
}

Status RunGraph(const std::shared_ptr<backend::ge_backend::GraphRunner> &runner, const RunOptions &options,
                const std::vector<GeTensorPtr> &inputs, std::vector<GeTensorPtr> *outputs) {
  MS_EXCEPTION_IF_NULL(runner);
  return runner->RunGraph(options, inputs, outputs);
}

Status RunGraphAsync(const std::shared_ptr<GraphRunner> &runner, const RunOptions &options,
                     const std::vector<GeTensorPtr> &inputs, std::vector<GeTensorPtr> *outputs) {
  MS_EXCEPTION_IF_NULL(runner);
  return runner->RunGraphAsync(options, inputs, outputs);
}

Status RunGraphWithStreamAsync(const std::shared_ptr<GraphRunner> &runner, const RunOptions &options, void *stream,
                               const std::vector<GeTensor> &inputs, std::vector<GeTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(runner);
  return runner->RunGraphWithStreamAsync(options, stream, inputs, outputs);
}

Status RegisterExternalAllocator(const std::shared_ptr<GraphRunner> &runner, const void *const stream,
                                 GeAllocatorPtr allocator) {
  MS_EXCEPTION_IF_NULL(runner);
  return runner->RegisterExternalAllocator(stream, allocator);
}

Status UnregisterExternalAllocator(const std::shared_ptr<GraphRunner> &runner, const void *const stream) {
  MS_EXCEPTION_IF_NULL(runner);
  return runner->UnregisterExternalAllocator(stream);
}

string ExportDFGraph(const std::string &file_name, const std::string &graph_name, bool is_save_to_file) {
  auto graph_runner = backend::ge_backend::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }
  return graph_runner->ExportDFGraph(file_name, graph_name, is_save_to_file);
}
}  // namespace mindspore::backend::ge_backend
