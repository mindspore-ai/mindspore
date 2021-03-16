/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CONNECTOR_SIZE_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CONNECTOR_SIZE_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "minddata/dataset/engine/perf/profiling.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"

using json = nlohmann::json;

namespace mindspore {
namespace dataset {
class ExecutionTree;

// Connector size sampling samples the output connector size of each op in the pipeline.
// It support JSON serialization for external usage.
class ConnectorSize : public Sampling {
  // Connector size sampling data is stored as a 2D vector
  //            op_0            ...         op_m
  // sample_0   size_0_0        ...         size_m_0
  // ...        ...             ...         ...
  // sample_n   size_0_m        ...         size_m_n
  //
  // A circular buffer will be implemented in the future to make this table more flexible.
  using ConnectorSizeSample = std::vector<int>;
  using ConnectorSizeSampleTable = std::vector<ConnectorSizeSample>;

 public:
  explicit ConnectorSize(ExecutionTree *tree) : tree_(tree) {}

  ~ConnectorSize() override = default;

  // Driver function for connector size sampling.
  // This function samples the connector size of every nodes within the ExecutionTree
  Status Sample() override;

  std::string Name() const override { return kConnectorSizeSamplingName; }

  // Save sampling data to file
  // @return Status The status code returned
  Status SaveToFile() override;

  Status Init(const std::string &dir_path, const std::string &device_id) override;

  // Parse op information and transform to json format
  json ParseOpInfo(const DatasetOp &node, const std::vector<int32_t> &size);

  // Change file mode after save throughput data
  Status ChangeFileMode() { return Status::OK(); }

  Status Analyze() override;

 private:
  ExecutionTree *tree_ = nullptr;          // ExecutionTree pointer
  ConnectorSizeSampleTable sample_table_;  // Dataset structure to store all samples of connector size sampling
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CONNECTOR_SIZE_H
