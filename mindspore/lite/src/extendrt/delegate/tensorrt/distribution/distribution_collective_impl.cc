/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/distribution/distribution_collective.h"
#include <unistd.h>
#include <thread>
#include <string>
#include "plugin/device/gpu/hal/device/distribution/collective_wrapper.h"
#include "src/extendrt/delegate/tensorrt/distribution/distribution_utils.h"
#include "src/extendrt/delegate/tensorrt/distribution/distribution_base.h"

namespace mindspore::lite {
DistributionCollective::DistributionCollective() {
  InitMPI();
  InitNCCLComm();
}

DistributionCollective &DistributionCollective::instance() {
  static DistributionCollective instance;
  return instance;
}

int DistributionCollective::ReduceScatterWrapper(const void *input_addr, void *output_addr, size_t count,
                                                 nvinfer1::DataType data_type, ReduceMode reduce_type,
                                                 cudaStream_t stream, const std::string &group) {
  int rank_id = GetRankID();
  MS_LOG(DEBUG) << "ReduceScatter on rank: " << rank_id;
  ncclResult_t ret = ReduceScatter(input_addr, output_addr, count, ConvertNCCLDataType(data_type),
                                   ConvertNCCLReduceMode(reduce_type), stream, group);
  if (ret != ncclSuccess) {
    MS_LOG(ERROR) << "ReduceScatter failed: " << static_cast<int>(ret);
    return RET_ERROR;
  }
  auto cuda_ret = cudaStreamSynchronize(stream);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaStreamSynchronize failed: " << static_cast<int>(cuda_ret);
    return RET_ERROR;
  }
  return RET_OK;
}

int DistributionCollective::AllGatherWrapper(const void *input_addr, void *output_addr, size_t count,
                                             nvinfer1::DataType data_type, cudaStream_t stream,
                                             const std::string &group_name) {
  int rank_id = GetRankID();
  MS_LOG(DEBUG) << "AllGather on rank: " << rank_id;
  ncclResult_t ret = AllGather(input_addr, output_addr, count, ConvertNCCLDataType(data_type), stream, group_name);
  if (ret != ncclSuccess) {
    MS_LOG(ERROR) << "AllGather failed: " << static_cast<int>(ret);
    return RET_ERROR;
  }
  auto cuda_ret = cudaStreamSynchronize(stream);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaStreamSynchronize failed: " << static_cast<int>(cuda_ret);
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
