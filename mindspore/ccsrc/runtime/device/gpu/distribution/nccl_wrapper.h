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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_NCCL_WRAPPER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_NCCL_WRAPPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>
#include <string>
#include <vector>
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
  void InitNCCLComm();
  ncclResult_t AllReduce(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                         ncclRedOp_t op, cudaStream_t stream, const std::string &group_name);
  ncclResult_t AllGather(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                         cudaStream_t stream, const std::string &group_name);
  ncclResult_t ReduceScatter(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                             ncclRedOp_t op, cudaStream_t stream, const std::string &group_name);
  ncclResult_t Broadcast(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type, int root,
                         cudaStream_t stream, const std::string &group_name);
  ncclResult_t Send(const void *send_addr, size_t count, ncclDataType_t data_type, int peer_rank, cudaStream_t stream,
                    const std::string &group_name);
  ncclResult_t Recv(void *recv_addr, size_t count, ncclDataType_t data_type, int peer_rank, cudaStream_t stream,
                    const std::string &group_name);
  ncclResult_t GroupStart();
  ncclResult_t GroupEnd();
  void AddGroupInfo(const std::string &group_name, NcclGroupInfo *group);
  void DestroyGroup(const std::string &group_name);
  std::vector<int> GetGroupRanks(const std::string &group_name);

 private:
  NCCLWrapper() : comm_init_done_(false) {}
  ~NCCLWrapper() = default;

 private:
  bool comm_init_done_;
  std::map<std::string, NcclGroupInfo> group_info_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_NCCL_WRAPPER_H_
