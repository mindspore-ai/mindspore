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

#ifndef MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_UTILS_H_
#include <string>
#include <map>
#include <memory>
#include <vector>
#include "transform/graph_ir/convert.h"
#include "transform/graph_ir/graph_runner.h"
#include "include/transform/graph_ir/types.h"
#include "transform/graph_ir/op_adapter_base.h"
#include "include/common/utils/config_manager.h"
#include "include/backend/visible.h"

namespace mindspore {
constexpr char BROADCAST_GRAPH_NAME[] = "broadcast_subgraph";
namespace transform {
using OpAdapterPtr = std::shared_ptr<transform::BaseOpAdapter>;
using GraphRunnerPtr = std::shared_ptr<transform::GraphRunner>;
using DfGraphConvertorPtr = std::shared_ptr<transform::DfGraphConvertor>;
OpAdapterPtr FindAdapter(const std::string &op_name, bool train = false);
OpAdapterPtr FindAdapter(AnfNodePtr node, bool train = false);

bool IsPartialSuccNode(const AnfNodePtr node);
bool IsWhileNode(const AnfNodePtr &node);
bool IsIfNode(const AnfNodePtr &node);
bool IsCaseNode(const AnfNodePtr &node);
std::string GetCNodeTargetFuncName(const CNodePtr cnode);
bool IsPartialCNode(const AnfNodePtr node);

void ClearGeSessionAndRunner();
// convert_type
std::vector<GeTensorPtr> ConvertInputTensors(const std::vector<MeTensorPtr> &me_tensors, const std::string &format);
std::vector<MeTensorPtr> ConvertGeTensors(const std::vector<GeTensorPtr> &ge_tensors);
GeDataType ConvertDataType(const MeDataType &type);

MeTensorPtr ConvertGeTensor(const GeTensorPtr &ge_tensor, const ShapeVector &request_dims);
MeTensorPtr ConvertGeTensor(const GeTensorPtr &tensor);
MeTensorPtr ConvertGeTensor(const GeTensorPtr &tensor, const TypeId &me_type);

// df graph manager
std::shared_ptr<transform::GraphRunner> GetGraphRunner();
BACKEND_EXPORT std::shared_ptr<ge::Session> GetGeSession();
BACKEND_EXPORT void SetGeSession(const std::shared_ptr<ge::Session> &sess_ptr);
BACKEND_EXPORT GraphRunnerPtr NewGraphRunner(const GraphRunnerOptions &options);
BACKEND_EXPORT void SetGraphRunner(const GraphRunnerPtr &runner);
BACKEND_EXPORT void ClearGraph();
BACKEND_EXPORT Status AddGraph(const std::string &name, const DfGraphPtr &graph, const OptionMap &options = {});
BACKEND_EXPORT void SetAnfGraph(const std::string &name, const AnfGraphPtr &anf_graph_ptr);
BACKEND_EXPORT DfGraphWrapperPtr GetGraphByName(const std::string &name);

FuncGraphPtr GetAnfGraph(uint32_t graph_id);

// convert
BACKEND_EXPORT DfGraphConvertorPtr NewConverter(const FuncGraphPtr &graph);

BACKEND_EXPORT void SetTraining(const DfGraphConvertorPtr &converter, bool training);
BACKEND_EXPORT void BuildGraph(const DfGraphConvertorPtr &converter,
                               const std::map<std::string, std::shared_ptr<tensor::Tensor>> &maps);
void GenerateBroadcastGraph(const DfGraphConvertorPtr &converter, const TensorOrderMap &tensors);
BACKEND_EXPORT void GenerateCheckpointGraph(const DfGraphConvertorPtr &converter);
BACKEND_EXPORT int ErrCode(const DfGraphConvertorPtr &converter);

BACKEND_EXPORT DfGraphPtr GetComputeGraph(const DfGraphConvertorPtr &converter);
BACKEND_EXPORT DfGraphPtr GetInitGraph(const DfGraphConvertorPtr &converter);
BACKEND_EXPORT DfGraphPtr GetSaveCheckpointGraph(const DfGraphConvertorPtr &converter);
BACKEND_EXPORT DfGraphPtr GetBroadcastGraph(const DfGraphConvertorPtr &converter);

// new session
BACKEND_EXPORT std::shared_ptr<ge::Session> NewSession(const SessionOptions &sess_options);

Status RunGraph(const std::shared_ptr<GraphRunner> &runner, const RunOptions &options,
                const std::vector<GeTensorPtr> &inputs, std::vector<GeTensorPtr> *outputs);

Status RunGraphAsync(const std::shared_ptr<GraphRunner> &runner, const RunOptions &options,
                     const std::vector<GeTensorPtr> &inputs, std::vector<GeTensorPtr> *outputs);

transform::Status CompileDatasetGraph(const DatasetGraphParam &param, const std::string &phase = "dataset");
}  // namespace transform
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_UTILS_H_
