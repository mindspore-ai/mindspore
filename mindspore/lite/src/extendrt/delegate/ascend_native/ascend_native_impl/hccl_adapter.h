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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_HCCL_ADAPTER
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_HCCL_ADAPTER
#include "hccl/hccl.h"
#include <hccl/hccl_types.h>
#include <vector>
namespace mindspore {
class HcclAdapter {
 public:
  const uint32_t HCCL_ROOT_INFO_BYTES = 4108;  // 4108: root info length
  const int root_rank = 0;
  HcclAdapter();
  explicit HcclAdapter(const HcclAdapter &hccl) = delete;
  virtual ~HcclAdapter();
  static HcclAdapter &GetInstance();
  int get_device() { return dev_id_; }
  int get_rank() { return rank_id_; }
  int get_size() { return rank_size_; }
  HcclResult Sync();
  HcclResult HcclInit();
  HcclResult AllGather(void *send_buff, void *recv_buff, uint64_t count, HcclDataType dataType, void *stream);
  HcclResult AllSumReduce(void *send_buff, void *recv_buff, uint64_t count, void *stream);
  void test_reduce(void *stream);
  HcclResult HcclSync(void *stream);

 private:
  HcclResult get_mpi_proc();
  HcclResult set_device_sat_mode();
  std::vector<int> getAviDevs(const char *devs);
  int dev_id_ = -1;
  int rank_id_ = 0;
  int rank_size_ = 1;
  HcclComm hccl_comm_ = nullptr;
  HcclRootInfo comm_id_;
};

};  // namespace mindspore
#endif
