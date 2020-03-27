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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_TF_CLIENT_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_TF_CLIENT_H_

#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "./example.pb.h"
#include "dataset/engine/datasetops/source/storage_client.h"
#include "dataset/util/status.h"

struct FileInfo {
  std::string fileName;
  uint64_t startRecordIdx;
  uint64_t endRecordIdx;
  uint64_t startOffset;
};

using QFile = std::queue<FileInfo>;

namespace mindspore {
namespace dataset {
// forward declares
class DataSchema;
class ParallelOp;

class TFClient : public StorageClient {
 public:
  // Name: Constructor
  // Description: Creates the TFClient.
  TFClient(std::unique_ptr<DataSchema> schema,  // In: The schema for this storage client.
           StorageOp *so);                      // In: The ParallelOp that's using this client

  ~TFClient() {}

  Status Init() override;

  // Name: Print()
  // Description: A function that prints info about the TFClient
  void Print(std::ostream &out) const override;  // In: The output stream to print to

  std::vector<uint64_t> ParseTfFileLines(const std::string &filename);

  Status ParseTfFileSchema(const std::string &filename);

  Status NextFileInfo(uint32_t id, FileInfo *);

  bool IsMoreData(uint32_t id) override;

  // Name: Reset()
  // Description: Resets any state info inside the client back to it's initialized
  //              state.
  Status Reset() override;

  Status ScatterFileRows(uint32_t device_id, const std::string &shard_config, uint32_t seed, bool shuffle_config);

 private:
  // hardcoded, put this in json schema
  // const static int32_t BERT_DATASET_TOTAL_ROWS = 43900;
  uint32_t rows_per_buffer_;
  std::default_random_engine random_seed_generator_;
  std::uniform_int_distribution<uint32_t> random_seed_distribution_;

  std::vector<std::pair<std::string, std::vector<uint64_t>>> v_file_rows_;
  std::vector<std::pair<std::string, std::vector<uint64_t>>> v_total_file_rows_;
  std::vector<QFile> f_info_queue_;
  uint64_t rows_per_shard_;
  std::vector<std::pair<uint32_t, uint32_t>> file_start_end_offset_;

  void InitStateInfo();

  std::vector<std::pair<std::string, std::vector<uint64_t>>> ShuffleVector(
    std::vector<std::pair<std::string, std::vector<uint64_t>>> v, uint32_t seed);

  Status CalculateRowsPerDevice();

  bool ValidFileForShard(const uint64_t file_rows, uint64_t *start_offset, uint64_t *end_offset,
                         const uint64_t &pre_count, uint32_t device_id) const;

  void CalculateNumRows();

  void GetValidFileForShard(const std::vector<std::pair<std::string, std::vector<uint64_t>>> &v_files,
                            uint32_t device_id);

  void CalculateStartOffset(const uint64_t start_index, const uint64_t end_index,
                            const std::vector<uint64_t> &vec_length, uint64_t *start_offset) const;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_TF_CLIENT_H_
