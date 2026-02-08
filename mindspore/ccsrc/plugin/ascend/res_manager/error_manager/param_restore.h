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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_SPECIAL_PARAMETER_REPLICATION_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_SPECIAL_PARAMETER_REPLICATION_H_

#include <limits>
#include <vector>
#include "hccl/hccl_types.h"
#include "plugin/ascend/res_manager/ascend_res_manager.h"
#include "plugin/ascend/res_manager/visible.h"

namespace mindspore {
namespace device {
namespace ascend {
// data structure for exchanging free device memory and parameter info between two nodes
class DataExchangeInfo {
 public:
  explicit DataExchangeInfo(const size_t num_params) { data_.resize(kDataExchangeInfoHeadSize + num_params); }
  DataExchangeInfo(const std::vector<tensor::TensorPtr> &params, size_t device_free_size);

  bool IsParamInfoSame(const DataExchangeInfo &other) {
    if (data_.size() != other.data_.size()) {
      return false;
    }
    // NOTE: data_[0] is free device memory size
    for (size_t i = 1; i < data_.size(); ++i) {
      if (data_[i] != other.data_[i]) {
        return false;
      }
    }
    return true;
  }

  uint64_t GetFreeDevMem() const { return data_[0]; }
  uint64_t GetNumOfParams() const { return data_[1]; }
  uint64_t GetSizeOfParam(int param_idx) { return data_[kDataExchangeInfoHeadSize + param_idx]; }
  uint64_t *GetData() { return data_.data(); }
  uint64_t GetSize() const { return data_.size(); }
  uint64_t GetSizeSum() const { return size_sum_; }
  uint64_t GetSizeMax() const { return size_max_; }

  static constexpr uint64_t kInvalidParamSize = std::numeric_limits<uint64_t>::max();

 private:
  // freeDeviceMemory and numberOfParameters
  static constexpr size_t kDataExchangeInfoHeadSize = 2;
  // free device memory and parameter info
  std::vector<uint64_t> data_;
  // sum of aligned parameter size
  uint64_t size_sum_ = 0;
  // max of aligned parameter size
  uint64_t size_max_ = 0;
};

class ASCEND_RES_MANAGER_EXPORT ParamReplication {
 public:
  explicit ParamReplication(const AscendResManager *res_mgr) : res_mgr_(res_mgr) {}
  ~ParamReplication() = default;

  void Init();

  int SendRecv(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank, bool use_batch);

 private:
  int DoParamInfoExchange(DataExchangeInfo *local_info, DataExchangeInfo *remote_info, int src_rank, int dst_rank);
  int CopyParamsInBatches(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank, void *xchg_buf_addr,
                          size_t xchg_buf_size);
  int CopyParamsOneByOne(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank);

 private:
  const AscendResManager *res_mgr_;
  size_t stream_id_ = 0;
  aclrtStream stream_ = nullptr;
  HcclComm comm_ = nullptr;
  int rank_id_ = -1;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_SPECIAL_PARAMETER_REPLICATION_H_
