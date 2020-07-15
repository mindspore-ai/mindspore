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

#ifndef MINDSPORE_CCSRC_DEVICE_GPU_DISTRIBUTION_NCCL_WRAPPER_H_
#define MINDSPORE_CCSRC_DEVICE_GPU_DISTRIBUTION_NCCL_WRAPPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>
#include <string>
#include <map>
#include "runtime/device/gpu/distribution/collective_common.h"

namespace mindspore {
namespace device {
namespace gpu {
class NCCLWrapper {
 public:
  NCCLWrapper(NCCLWrapper const &) = delete;
  NCCLWrapper &operator=(const NCCLWrapper &) = delete;
  static NCCLWrapper &instance();
  ncclUniqueId nccl_unique_id() const;
  void set_nccl_unique_id(ncclUniqueId unique_id);
  void set_rank(int rank_id, int rank_size);
  void InitNCCLComm();
  void InitNCCLComm(ncclComm_t *comm, int rank_size, ncclUniqueId unique_id, int rank);
  ncclResult_t AllReduce(const void *input_addr, void *output_addr, size_t count, ncclDataType_t datatype,
                         ncclRedOp_t op, cudaStream_t stream, const std::string &group_name = NCCL_WORLD_GROUP);
  ncclResult_t AllGather(const void *input_addr, void *output_addr, size_t count, ncclDataType_t datatype,
                         cudaStream_t stream, const std::string &group_name = NCCL_WORLD_GROUP);
  ncclResult_t ReduceScatter(const void *input_addr, void *output_addr, size_t count, ncclDataType_t datatype,
                             ncclRedOp_t op, cudaStream_t stream, const std::string &group_name = NCCL_WORLD_GROUP);
  void SetGroupNameToNCCLComm(const std::string &group_name, const ncclComm_t comm);
  void DestroyGroup(const std::string &group_name);

 private:
  NCCLWrapper() : rank_id_(-1), rank_size_(0) {}
  ~NCCLWrapper() = default;

 private:
  int rank_id_;
  int rank_size_;
  ncclUniqueId unique_id_;
  ncclComm_t comm_;
  std::map<std::string, ncclComm_t> group_to_comm_map_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_GPU_DISTRIBUTION_NCCL_WRAPPER_H_
