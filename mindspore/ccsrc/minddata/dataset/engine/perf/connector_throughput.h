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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CONNECTOR_THROUGHPUT_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CONNECTOR_THROUGHPUT_H

#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include "minddata/dataset/engine/perf/profiling.h"
#include "minddata/dataset/engine/perf/perf_data.h"
#include "minddata/dataset/engine/perf/cyclic_array.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/execution_tree.h"

using json = nlohmann::json;
namespace mindspore {
namespace dataset {
// Connector throughput samples the output connector size of each op in the pipeline.
// For the description of the data structure see perf_buffer.h
// It support JSON serialization for external usage.
class ConnectorThroughput : public Sampling {
  using OutBufferCount = PerfData<CyclicArray<int64_t>>;
  using Throughput = PerfData<CyclicArray<double>>;
  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;
  using TimeStamps = PerfData<CyclicArray<TimePoint>>;

 public:
  explicit ConnectorThroughput(ExecutionTree *tree, int64_t max_rows = 1000000)
      : tree_(tree),
        max_rows_(max_rows),
        n_nodes_(InitNodes()),
        out_buffer_count_table_(OutBufferCount(max_rows_, n_nodes_)),
        throughput_(Throughput(max_rows_, n_nodes_)),
        timestamps_(TimeStamps(max_rows_, 1)) {
    timestamps_.AddSample(std::vector<TimePoint>(1));
    out_buffer_count_table_.AddSample(std::vector<int64_t>(n_nodes_));
  }

  /// \brief Destructor
  ~ConnectorThroughput() = default;

  // Driver function for connector size sampling.
  // This function samples the connector size of every nodes within the ExecutionTree
  Status Sample() override;

  // Traverse the tree nodes and count them
  int InitNodes();

  std::string Name() const override { return name_; };

  // Save sampling data to file
  // @return Status The status code returned
  Status SaveToFile() override;

  Status Init(const std::string &dir_path, const std::string &device_id) override;

  json ParseOpInfo(const DatasetOp &node, const std::vector<double> &thr);

  Status ChangeFileMode() override;

  Status Analyze() override;

 private:
  ExecutionTree *tree_ = nullptr;  // ExecutionTree pointer
  int64_t max_rows_;
  int32_t n_nodes_;
  OutBufferCount out_buffer_count_table_;
  Throughput throughput_;
  TimeStamps timestamps_;
  std::string name_ = kConnectorThroughputSamplingName;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CONNECTOR_THROUGHPUT_H
