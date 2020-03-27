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

#include "device/gpu/distribution/nccl_wrapper.h"

namespace mindspore {
namespace device {
namespace gpu {
NCCLWrapper &NCCLWrapper::instance() {
  static NCCLWrapper instance;
  return instance;
}

ncclUniqueId NCCLWrapper::nccl_unique_id() const {
  ncclUniqueId unique_id;
  CHECK_RET(ncclGetUniqueId(&unique_id), ncclSuccess, "Failed to create nccl unique id.");
  return unique_id;
}

void NCCLWrapper::set_nccl_unique_id(ncclUniqueId unique_id) { unique_id_ = unique_id; }

void NCCLWrapper::set_rank(int rank_id, int rank_size) {
  rank_id_ = rank_id;
  rank_size_ = rank_size;
}

void NCCLWrapper::InitNCCLComm() {
  CHECK_RET(ncclCommInitRank(&comm_, rank_size_, unique_id_, rank_id_), ncclSuccess,
            "Failed to init nccl communicator.");
}

ncclResult_t NCCLWrapper::AllReduce(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                    ncclRedOp_t reduce_type, cudaStream_t stream) {
  return ncclAllReduce(input_addr, output_addr, count, data_type, reduce_type, comm_, stream);
}

ncclResult_t NCCLWrapper::AllGather(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                    cudaStream_t stream) {
  return ncclAllGather(input_addr, output_addr, count, data_type, comm_, stream);
}

ncclResult_t NCCLWrapper::ReduceScatter(const void *input_addr, void *output_addr, size_t count,
                                        ncclDataType_t data_type, ncclRedOp_t reduce_type, cudaStream_t stream) {
  return ncclReduceScatter(input_addr, output_addr, count, data_type, reduce_type, comm_, stream);
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
