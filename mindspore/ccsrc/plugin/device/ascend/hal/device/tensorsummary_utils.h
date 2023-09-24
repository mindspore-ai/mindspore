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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORSUMMARY_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORSUMMARY_UTILS_H_

#include <map>
#include <string>
#include <thread>
#include <functional>
#include "acl/acl_tdt.h"
#include "ir/tensor.h"
#include "plugin/device/ascend/hal/device/tensorprint_utils.h"

namespace mindspore::device::ascend {

struct TDTInfo {
  std::string channel_name;
  acltdtChannelHandle *acl_handle;
  ChannelType channel_type;
  std::thread *dtd_thread;
};

class TensorSummaryUtils {
 public:
  static TensorSummaryUtils &GetInstance();
  void CreateTDTSummaryThread();
  void DestroyTDTSummaryThread();

 private:
  static void GetSummaryData(string channel_name, acltdtChannelHandle *acl_handle);
};

class TDTTensorUtils {
 public:
  static TDTTensorUtils &GetInstance();
  acltdtChannelHandle *CreateChannel(std::string name, ChannelType *channel_type);
  void ReceiveData(std::string channel_name, const acltdtChannelHandle *acl_handle);

  std::map<std::string, TDTInfo> tdt_infos;

 private:
  bool ConvertDataset2Tensor(acltdtDataset *acl_dataset, ChannelType channel_type, const char *summary_name);
  bool PrintTensorToString(const char *str_data_ptr, mindspore::tensor::Tensor *print_tensor,
                           const size_t &memory_size);
  std::string TensorNameToSummaryName(std::string channel_name, std::string tensor_name);
};
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORSUMMARY_UTILS_H_
