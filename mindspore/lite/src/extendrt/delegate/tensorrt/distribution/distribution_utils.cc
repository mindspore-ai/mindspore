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

#include "src/extendrt/delegate/tensorrt/distribution/distribution_utils.h"
#include <unordered_map>
#include "src/common/log_adapter.h"

namespace mindspore::lite {
ncclDataType_t ConvertNCCLDataType(nvinfer1::DataType type_id) {
  std::unordered_map<nvinfer1::DataType, ncclDataType_t> data_type_map = {
    {nvinfer1::DataType::kINT8, ncclInt8},
    {nvinfer1::DataType::kINT32, ncclInt32},
    {nvinfer1::DataType::kFLOAT, ncclFloat32},
    {nvinfer1::DataType::kHALF, ncclHalf},
  };
  auto iter = data_type_map.find(type_id);
  ncclDataType_t data_type;
  if (iter != data_type_map.end()) {
    data_type = iter->second;
  } else {
    data_type = ncclFloat32;
    MS_LOG(WARNING) << "invalid data_type for NCCL, need check: " << static_cast<int>(type_id);
  }
  return data_type;
}

ncclRedOp_t ConvertNCCLReduceMode(ReduceMode mode) {
  std::unordered_map<ReduceMode, ncclRedOp_t> reduce_ops_ = {
    // higher version support mean {schema::ReduceMode::ReduceMode_ReduceMean, ncclAvg},
    {ReduceMode::Reduce_Max, ncclMax},
    {ReduceMode::Reduce_Min, ncclMin},
    {ReduceMode::Reduce_Prod, ncclProd},
    {ReduceMode::Reduce_Sum, ncclSum},
  };
  auto iter = reduce_ops_.find(mode);
  ncclRedOp_t nccl_mode;
  if (iter != reduce_ops_.end()) {
    nccl_mode = iter->second;
  } else {
    nccl_mode = ncclSum;
    MS_LOG(WARNING) << "invalid reduce for NCCL, need check: " << static_cast<int>(mode);
  }
  return nccl_mode;
}
}  // namespace mindspore::lite
