/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_SHARED_MEMORY_QUEUE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_SHARED_MEMORY_QUEUE_H_

#include <memory>
#include <utility>
#include <vector>
#if !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID)
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <errno.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <stdlib.h>
#include <sys/shm.h>
#endif

#include "include/api/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
#if !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID)
class DATASET_API SharedMemoryQueue {
 public:
  explicit SharedMemoryQueue(const key_t &key);

  ~SharedMemoryQueue();

  // Convert TensorRow to shared memory
  // The shared memory format like below:
  // flag, uint32_t, the flag maybe kFlagNone, kFlagEOE, kFlagEOF, kFlagWait, kFlagQuit, kFlagSkip, kFlagError
  // size, uint32_t, the size of tensor in the TensorRow
  //        types, [uint32_t, uint32_t, uint32_t, ...], the type of the Tensor which maybe:
  //                                                    0: data_
  //                                                    1: python_array_
  //                                                    2: python_dict_
  //        shapes, [uint32_t, [], uint32_t, [], uint32_t, [], ...], every shape of the Tensor
  //        types, [uint32_t , uint32_t, uint32_t, ...], the data type of the Tensor
  //        data, [length, data, length, data, length, data, ...], the data of the Tensor
  //                                                               length, uint64_t
  //                                                               data, char, the memory data
  Status FromTensorRow(const TensorRow &in_row);

  Status ToTensorRow(TensorRow *out_row, const int &shm_id, const uint64_t &shm_size);

  Status ToTensorRowWithNoCopy(TensorRow *out_row);

  void SetReleaseFlag(bool flag);

  key_t GetKey();

  int GetShmID();

  uint64_t GetShmSize();

 private:
  Status ReleaseCurrentShm();

  Status CreateShmBySize(const uint64_t &size);

  Status UpdateShmBySize(const uint64_t &size);

  Status CalculateShmSize(const TensorRow &in_row, uint64_t *size);

  Status Serialize(const TensorRow &in_row);

  Status Deserialize(TensorRow *in_row);

 private:
  key_t key_;          // the shm key
  int shm_id_;         // the shm id
  void *shm_addr_;     // the shm addr
  uint64_t shm_size_;  // the shm size
  bool release_flag_;  // whether release the shm when deconstruct
};
#endif
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_SHARED_MEMORY_QUEUE_H_
