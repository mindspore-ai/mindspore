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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_SHARED_MEMORY_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_SHARED_MEMORY_H_

#include <sys/ipc.h>
#include <sys/shm.h>
#include <mutex>
#include <string>

#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
namespace gnn {

const int kGnnSharedMemoryId = 65;

class GraphSharedMemory {
 public:
  explicit GraphSharedMemory(int64_t memory_size, key_t memory_key);
  explicit GraphSharedMemory(int64_t memory_size, const std::string &mr_file);

  ~GraphSharedMemory();

  // @param uint8_t** shared_memory - shared memory address
  // @return Status - the status code
  Status CreateSharedMemory();

  // @param uint8_t** shared_memory - shared memory address
  // @return Status - the status code
  Status GetSharedMemory();

  Status DeleteSharedMemory();

  Status InsertData(const uint8_t *data, int64_t len, int64_t *offset);

  Status GetData(uint8_t *data, int64_t data_len, int64_t offset, int64_t get_data_len);

  key_t memory_key() { return memory_key_; }

  int64_t memory_size() { return memory_size_; }

 private:
  Status SharedMemoryImpl(const int &shmflg);

  std::string mr_file_;
  int64_t memory_size_;
  key_t memory_key_;
  std::string memory_key_str_;
  uint8_t *memory_ptr_;
  int64_t memory_offset_;
  std::mutex mutex_;
  bool is_new_create_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_SHARED_MEMORY_H_
