/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORPRINT_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORPRINT_UTILS_H_

#include <memory>
#include <string>
#include "plugin/device/ascend/hal/device/mbuf_receive_manager.h"

namespace mindspore::device::ascend {
class TensorPrintUtils {
 public:
  static TensorPrintUtils &GetInstance();

  TensorPrintUtils();
  ~TensorPrintUtils();
  TensorPrintUtils(const TensorPrintUtils &) = delete;
  TensorPrintUtils &operator=(const TensorPrintUtils &) = delete;
  void PrintReceiveData(const ScopeAclTdtDataset &dataset);

 private:
  void OutputReceiveData2PbFile(const ScopeAclTdtDataset &dataset);

  std::string print_file_path_;
  // stream of output file in protobuf binary format
  std::shared_ptr<std::fstream> pb_file_stream_{nullptr};
};
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORPRINT_UTILS_H_
