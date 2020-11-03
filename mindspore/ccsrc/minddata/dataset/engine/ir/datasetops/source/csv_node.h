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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_CSV_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_CSV_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {
/// \brief Record type for CSV
enum CsvType : uint8_t { INT = 0, FLOAT, STRING };

/// \brief Base class of CSV Record
class CsvBase {
 public:
  CsvBase() = default;
  explicit CsvBase(CsvType t) : type(t) {}
  virtual ~CsvBase() {}
  CsvType type;
};

/// \brief CSV Record that can represent integer, float and string.
template <typename T>
class CsvRecord : public CsvBase {
 public:
  CsvRecord() = default;
  CsvRecord(CsvType t, T v) : CsvBase(t), value(v) {}
  ~CsvRecord() {}
  T value;
};

class CSVNode : public DatasetNode {
 public:
  /// \brief Constructor
  CSVNode(const std::vector<std::string> &dataset_files, char field_delim,
          const std::vector<std::shared_ptr<CsvBase>> &column_defaults, const std::vector<std::string> &column_names,
          int64_t num_samples, ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
          std::shared_ptr<DatasetCache> cache);

  /// \brief Destructor
  ~CSVNode() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return shared pointer to the list of newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  /// \brief Get the shard id of node
  /// \return Status Status::OK() if get shard id successfully
  Status GetShardId(int32_t *shard_id) override;

 private:
  std::vector<std::string> dataset_files_;
  char field_delim_;
  std::vector<std::shared_ptr<CsvBase>> column_defaults_;
  std::vector<std::string> column_names_;
  int64_t num_samples_;
  ShuffleMode shuffle_;
  int32_t num_shards_;
  int32_t shard_id_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_CSV_NODE_H_
