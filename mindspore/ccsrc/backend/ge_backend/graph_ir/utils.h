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

#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GRAPH_IR_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GRAPH_IR_UTILS_H_
#include <string>
#include <map>
#include <memory>
#include <vector>
#include "backend/ge_backend/graph_ir/convert.h"
#include "backend/ge_backend/graph_ir/graph_runner.h"
#include "backend/ge_backend/graph_ir/types.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "include/backend/visible.h"

namespace mindspore {
constexpr char BROADCAST_GRAPH_NAME[] = "broadcast_subgraph";

namespace backend::ge_backend {
using OpAdapterPtr = std::shared_ptr<device::ascend::BaseOpAdapter>;
using GraphRunnerPtr = std::shared_ptr<backend::ge_backend::GraphRunner>;
using DfGraphConvertorPtr = std::shared_ptr<backend::ge_backend::DfGraphConvertor>;

bool IsInitDataSetQueueNode(const AnfNodePtr &node);

BACKEND_EXPORT void ClearGeSessionAndRunner();
BACKEND_EXPORT void InitializeAoeUtil(const std::string &aoe_job_type);
BACKEND_EXPORT void DestroyAoeUtil();
BACKEND_EXPORT void EnableAoeOffline();

// convert_type
std::vector<GeTensorPtr> ConvertInputTensors(const std::vector<MeTensorPtr> &me_tensors, const std::string &format);
std::vector<MeTensorPtr> ConvertGeTensors(const std::vector<GeTensorPtr> &ge_tensors);
GeDataType ConvertDataType(const MeDataType &type);

MeTensorPtr ConvertGeTensor(const GeTensorPtr &ge_tensor, const ShapeVector &request_dims, bool ref_mem = false);
MeTensorPtr ConvertGeTensor(const GeTensorPtr &tensor);
MeTensorPtr ConvertGeTensor(const GeTensorPtr &tensor, const TypeId &me_type);

// df graph manager
BACKEND_EXPORT std::shared_ptr<backend::ge_backend::GraphRunner> GetGraphRunner();
std::shared_ptr<backend::ge_backend::GraphRunner> CheckAndGetGraphRunner(
  const backend::ge_backend::RunOptions &run_options);
BACKEND_EXPORT std::shared_ptr<::ge::Session> GetGeSession();
BACKEND_EXPORT void SetGeSession(const std::shared_ptr<::ge::Session> &sess_ptr);
BACKEND_EXPORT GraphRunnerPtr NewGraphRunner(const GraphRunnerOptions &options);
BACKEND_EXPORT void SetGraphRunner(const GraphRunnerPtr &runner);
BACKEND_EXPORT void ClearGraph();
BACKEND_EXPORT Status AddGraph(const std::string &name, const DfGraphPtr &graph,
                               const DfGraphConfig &graph_config = DfGraphConfig({}, false, false, false));
void SetAnfGraph(const std::string &name, const AnfGraphPtr &anf_graph_ptr);
BACKEND_EXPORT DfGraphWrapperPtr GetGraphByName(const std::string &name);
void AddOptimizeGraph(const std::string &name);

FuncGraphPtr GetAnfGraph(uint32_t graph_id);

// convert
BACKEND_EXPORT DfGraphConvertorPtr NewConverter(const FuncGraphPtr &graph, const std::string &phase_prefix = "",
                                                RefModeFlag ref_mode_type = RefModeFlag::kRefModeEnv,
                                                bool offline_convert = false);

BACKEND_EXPORT void SetTraining(const DfGraphConvertorPtr &converter, bool training);
BACKEND_EXPORT void SetExportAir(const DfGraphConvertorPtr &converter, bool export_air);
BACKEND_EXPORT void BuildGraph(const std::string &name, const DfGraphConvertorPtr &converter,
                               const std::map<std::string, std::shared_ptr<tensor::Tensor>> &maps);
BACKEND_EXPORT void GenerateBroadcastGraph(const DfGraphConvertorPtr &converter, const TensorOrderMap &tensors);
BACKEND_EXPORT void GenerateCheckpointGraph(const DfGraphConvertorPtr &converter);
BACKEND_EXPORT int ErrCode(const DfGraphConvertorPtr &converter);
BACKEND_EXPORT void GenFakeGraph(const std::string &name, const DfGraphConvertorPtr &converter);

BACKEND_EXPORT DfGraphPtr GetComputeGraph(const DfGraphConvertorPtr &converter);
BACKEND_EXPORT DfGraphPtr GetInitGraph(const DfGraphConvertorPtr &converter);
DfGraphPtr GetSaveCheckpointGraph(const DfGraphConvertorPtr &converter);
BACKEND_EXPORT DfGraphPtr GetBroadcastGraph(const DfGraphConvertorPtr &converter);

// new session
BACKEND_EXPORT std::shared_ptr<::ge::Session> NewSession(const SessionOptions &sess_options);

BACKEND_EXPORT Status RunGraph(const std::shared_ptr<GraphRunner> &runner, const RunOptions &options,
                               const std::vector<GeTensorPtr> &inputs, std::vector<GeTensorPtr> *outputs);

Status RunGraphAsync(const std::shared_ptr<GraphRunner> &runner, const RunOptions &options,
                     const std::vector<GeTensorPtr> &inputs, std::vector<GeTensorPtr> *outputs);

BACKEND_EXPORT Status RunGraphWithStreamAsync(const std::shared_ptr<GraphRunner> &runner, const RunOptions &options,
                                              void *stream, const std::vector<GeTensor> &inputs,
                                              std::vector<GeTensor> *outputs);

BACKEND_EXPORT Status RegisterExternalAllocator(const std::shared_ptr<GraphRunner> &runner, const void *const stream,
                                                GeAllocatorPtr allocator);

BACKEND_EXPORT Status UnregisterExternalAllocator(const std::shared_ptr<GraphRunner> &runner, const void *const stream);

BACKEND_EXPORT string ExportDFGraph(const std::string &file_name, const std::string &graph_name, bool is_save_to_file);
}  // namespace backend::ge_backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GRAPH_IR_UTILS_H_
