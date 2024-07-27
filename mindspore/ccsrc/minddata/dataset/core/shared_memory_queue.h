/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <utility>
#include <vector>

#if !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID)
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>
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

const int kShmPermission = 0600;

// The following data type indicates the memory size occupied by TensorRow serialization.
const int kTensorRowType = 4;
const int kTensorSizeInTensorRow = 4;
const int kTensorType = 4;
const int kTensorShapeDims = 4;
const int kTensorShapeType = 4;
const int kTensorDataType = 4;
const int kTensorDataLen = 8;

// The following types represent the actual data types stored in the tensor.
const int kNormalCTensor = 0;
const int kPythonDictObject = 1;

class DATASET_API SharedMemoryQueue {
 public:
  explicit SharedMemoryQueue(const key_t &key);

  ~SharedMemoryQueue();

  // Convert TensorRow to shared memory
  // The shared memory format like below:
  // flag, uint32_t, the flag maybe kFlagNone, kFlagEOE, kFlagEOF, kFlagWait, kFlagQuit, kFlagSkip, kFlagError
  // size, uint32_t, the size of tensor in the TensorRow
  //        types, [uint32_t, uint32_t, uint32_t, ...], the type of the Tensor which maybe:
  //                                                    0: data_ / python_array_
  //                                                    1: python_dict_
  // case1: tensor is C Tensor with data_ / Python Array with data_
  //        shapes, [uint32_t, [], uint32_t, [], uint32_t, [], ...], every shape of the Tensor
  //        types, [uint32_t , uint32_t, uint32_t, ...], the data type of the Tensor
  //        data, [length, data, length, data, length, data, ...], the data of the Tensor
  //                                                               length, uint64_t
  //                                                               data, char, the memory data
  // case2: tensor is Python Dict wiht data_ but without shape & type
  //        data, [length, data, length, data, ...], the data of the Tensor
  Status FromTensorRow(const TensorRow &in_row);

  Status ToTensorRow(TensorRow *out_row, const int &shm_id, const uint64_t &shm_size);

  Status ToTensorRowWithNoCopy(TensorRow *out_row);

  void SetReleaseFlag(bool flag);

  key_t GetKey();

  int GetShmID();

  uint64_t GetShmSize();

  Status ReleaseCurrentShm();

 private:
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
