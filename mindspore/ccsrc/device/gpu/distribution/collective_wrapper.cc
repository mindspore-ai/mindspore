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

#include <mpi.h>
#include <nccl.h>
#include <unistd.h>
#include <memory>
#include <string>
#include <iostream>
#include "device/gpu/distribution/mpi_wrapper.h"
#include "device/gpu/distribution/nccl_wrapper.h"

#ifndef EXPORT_WRAPPER
#define EXPORT_WRAPPER __attribute__((visibility("default")))
#endif

using MPIWrapper = mindspore::device::gpu::MPIWrapper;
using NCCLWrapper = mindspore::device::gpu::NCCLWrapper;

extern "C" EXPORT_WRAPPER void InitMPI() { MPIWrapper::instance(); }

extern "C" EXPORT_WRAPPER int local_rank_id() { return MPIWrapper::instance().local_rank_id(); }

extern "C" EXPORT_WRAPPER void InitNCCLComm() { NCCLWrapper::instance().InitNCCLComm(); }

extern "C" EXPORT_WRAPPER ncclResult_t AllReduce(const void *input_addr, void *output_addr, size_t count,
                                                 ncclDataType_t data_type, ncclRedOp_t reduce_type,
                                                 cudaStream_t stream) {
  return NCCLWrapper::instance().AllReduce(input_addr, output_addr, count, data_type, reduce_type, stream);
}

extern "C" EXPORT_WRAPPER ncclResult_t AllGather(const void *input_addr, void *output_addr, size_t count,
                                                 ncclDataType_t data_type, cudaStream_t stream) {
  return NCCLWrapper::instance().AllGather(input_addr, output_addr, count, data_type, stream);
}

extern "C" EXPORT_WRAPPER ncclResult_t ReduceScatter(const void *input_addr, void *output_addr, size_t count,
                                                     ncclDataType_t data_type, ncclRedOp_t reduce_type,
                                                     cudaStream_t stream) {
  return NCCLWrapper::instance().ReduceScatter(input_addr, output_addr, count, data_type, reduce_type, stream);
}
