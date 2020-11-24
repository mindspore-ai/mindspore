/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_GPU_KERNEL_H_

#include <nccl.h>
#include <map>
#include <string>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "runtime/device/gpu/distribution/collective_init.h"

namespace mindspore {
namespace kernel {
static std::map<std::string, ncclDataType_t> kNcclDtypeMap = {
  {"kNumberTypeFloat32", ncclFloat}, {"kNumberTypeFloat16", ncclHalf}, {"kNumberTypeInt32", ncclInt}};

typedef ncclResult_t (*AllReduce)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, cudaStream_t,
                                  const std::string &);
typedef ncclResult_t (*AllGather)(const void *, void *, size_t, ncclDataType_t, cudaStream_t, const std::string &);
typedef ncclResult_t (*ReduceScatter)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, cudaStream_t,
                                      const std::string &);
typedef ncclResult_t (*Broadcast)(const void *, void *, size_t, ncclDataType_t, int, cudaStream_t, const std::string &);
typedef ncclResult_t (*Send)(const void *, size_t, ncclDataType_t, int, cudaStream_t, const std::string &);
typedef ncclResult_t (*Recv)(void *, size_t, ncclDataType_t, int, cudaStream_t, const std::string &);
typedef ncclResult_t (*GroupStart)();
typedef ncclResult_t (*GroupEnd)();
typedef std::vector<int> (*GetGroupRanks)(const std::string &);

class NcclGpuKernel : public GpuKernel {
 public:
  NcclGpuKernel() : group_name_(""), nccl_data_type_(ncclHalf) {}
  ~NcclGpuKernel() override = default;

 protected:
  ncclDataType_t nccl_dtype(const TypeId &type_id) { return kNcclDtypeMap[TypeIdLabel(type_id)]; }

  std::string group_name_;
  ncclDataType_t nccl_data_type_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_GPU_KERNEL_H_
