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
#include "include/common/visible.h"

namespace mindspore {
const char BROADCAST_GRAPH_NAME[] = "broadcast_subgraph";
namespace transform {
using OpAdapterPtr = std::shared_ptr<transform::BaseOpAdapter>;
using GraphRunnerPtr = std::shared_ptr<transform::GraphRunner>;
using DfGraphConvertorPtr = std::shared_ptr<transform::DfGraphConvertor>;
COMMON_EXPORT OpAdapterPtr FindAdapter(const std::string &op_name, bool train = false);
COMMON_EXPORT OpAdapterPtr FindAdapter(AnfNodePtr node, bool train = false);

COMMON_EXPORT bool IsPartialSuccNode(const AnfNodePtr node);
COMMON_EXPORT bool IsWhileNode(const AnfNodePtr &node);
COMMON_EXPORT std::string GetCNodeTargetFuncName(const CNodePtr cnode);
COMMON_EXPORT bool IsCaseNode(const CNodePtr node);
COMMON_EXPORT bool IsPartialCNode(const AnfNodePtr node);

COMMON_EXPORT void EraseGeResource();
COMMON_EXPORT void ClearGraphWrapper();
COMMON_EXPORT void ClearGeSessionAndRunner();
// convert_type
COMMON_EXPORT std::vector<GeTensorPtr> ConvertInputTensors(const std::vector<MeTensorPtr> &me_tensors,
                                                           const std::string &format);
COMMON_EXPORT std::vector<MeTensorPtr> ConvertGeTensors(const std::vector<GeTensorPtr> &ge_tensors);
COMMON_EXPORT GeDataType ConvertDataType(const MeDataType &type);

COMMON_EXPORT MeTensorPtr ConvertGeTensor(GeTensorPtr ge_tensor, const ShapeVector &request_dims);
COMMON_EXPORT MeTensorPtr ConvertGeTensor(const GeTensorPtr &tensor);
COMMON_EXPORT MeTensorPtr ConvertGeTensor(const GeTensorPtr &tensor, const TypeId &me_type);
// df graph manager
COMMON_EXPORT std::shared_ptr<transform::GraphRunner> GetGraphRunner();
COMMON_EXPORT std::shared_ptr<ge::Session> GetGeSession();
COMMON_EXPORT void SetGeSession(const std::shared_ptr<ge::Session> &sess_ptr);
COMMON_EXPORT GraphRunnerPtr NewGraphRunner(const GraphRunnerOptions &options);
COMMON_EXPORT void SetGraphRunner(const GraphRunnerPtr &runner);
COMMON_EXPORT void ClearGraph();
COMMON_EXPORT Status AddGraph(const std::string &name, const DfGraphPtr &graph, const OptionMap &options = {});
COMMON_EXPORT void SetAnfGraph(const std::string &name, const AnfGraphPtr &anf_graph_ptr);
COMMON_EXPORT DfGraphWrapperPtr GetGraphByName(const std::string &name);

COMMON_EXPORT FuncGraphPtr GetAnfGraph(uint32_t graph_id);

// convert
COMMON_EXPORT DfGraphConvertorPtr NewConverter(const FuncGraphPtr &graph);

COMMON_EXPORT void SetTraining(DfGraphConvertorPtr converter, bool training);
COMMON_EXPORT void BuildGraph(DfGraphConvertorPtr converter,
                              const std::map<std::string, std::shared_ptr<tensor::Tensor>> &maps);
COMMON_EXPORT void GenerateBroadcastGraph(DfGraphConvertorPtr converter, const TensorOrderMap &tensors);
COMMON_EXPORT void GenerateCheckpointGraph(DfGraphConvertorPtr converter);
COMMON_EXPORT int ErrCode(DfGraphConvertorPtr converter);
COMMON_EXPORT void DrawComputeGraph(DfGraphConvertorPtr converter, const std::string &name);
COMMON_EXPORT void DrawInitGraph(DfGraphConvertorPtr converter, const std::string &name);
COMMON_EXPORT void DrawSaveCheckpointGraph(DfGraphConvertorPtr converter, const std::string &name);
COMMON_EXPORT DfGraphPtr GetComputeGraph(DfGraphConvertorPtr converter);
COMMON_EXPORT DfGraphPtr GetInitGraph(DfGraphConvertorPtr converter);
COMMON_EXPORT DfGraphPtr GetSaveCheckpointGraph(DfGraphConvertorPtr converter);
COMMON_EXPORT DfGraphPtr GetBroadcastGraph(DfGraphConvertorPtr converter);

// new session
COMMON_EXPORT std::shared_ptr<ge::Session> NewSession(const SessionOptions &sess_options);

COMMON_EXPORT Status RunGraph(const std::shared_ptr<GraphRunner> &runner, const RunOptions &options,
                              const std::vector<GeTensorPtr> &inputs, std::vector<GeTensorPtr> *outputs);

COMMON_EXPORT Status RunGraph(const std::shared_ptr<GraphRunner> &runner, const RunOptions &options,
                              const std::vector<GeTensorPtr> &inputs, std::vector<MeTensorPtr> *outputs,
                              const std::vector<TypeId> &me_types);
COMMON_EXPORT void ClearOpAdapterMap();

COMMON_EXPORT transform::Status CompileDatasetGraph(const DatasetGraphParam &param,
                                                    const std::string &phase = "dataset");
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_UTILS_H_
