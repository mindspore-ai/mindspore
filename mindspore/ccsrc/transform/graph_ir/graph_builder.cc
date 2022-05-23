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

#include "transform/graph_ir/graph_builder.h"

#include <sstream>

#include "ops/math_ops.h"

namespace mindspore {
namespace transform {
DfGraphPtr BuildMDDatasetGraph(const DatasetGraphParam &param) {
  MS_LOG(INFO) << "BuildMDDatasetGraph.";

  // InitData
  auto d = ge::op::InitData("init_data_tmp").set_attr_channel_name(param.queue_name());

  // set graph inputs & outputs
  std::vector<ge::Operator> inputs{d};
  std::vector<ge::Operator> outputs{d};
  DfGraphPtr dataset_graph = std::make_shared<DfGraph>("dataset");
  (void)dataset_graph->SetInputs(inputs);
  (void)dataset_graph->SetOutputs(outputs);

  return dataset_graph;
}

Status BuildDatasetGraph(const DatasetGraphParam &param, const std::string &phase) {
  Status ret;
  std::string graph_name = phase;

  MS_LOG(INFO) << "BuildDatasetGraph begin. phase is " << phase;
  MS_LOG(INFO) << "param is " << param.ToString() << ".";

  DfGraphPtr dataset_graph = BuildMDDatasetGraph(param);
  ret = DfGraphManager::GetInstance().AddGraph(graph_name, dataset_graph);
  if (ret != Status::SUCCESS) {
    MS_LOG(ERROR) << "BuildDatasetGraph failed.";
  } else {
    MS_LOG(INFO) << "BuildDatasetGraph end.";
  }
  return ret;
}
}  // namespace transform
}  // namespace mindspore
