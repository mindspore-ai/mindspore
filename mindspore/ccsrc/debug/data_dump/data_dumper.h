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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_DATA_DUMPER_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_DATA_DUMPER_H_

#include <map>
#include <string>
#include <memory>
#include "include/backend/device_type.h"
#include "utils/log_adapter.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace datadump {

class BACKEND_EXPORT DataDumper {
 public:
  virtual void Initialize() { MS_LOG(WARNING) << "Initialize ACL DataDumper"; }
  virtual void EnableDump(uint32_t device_id, uint32_t step_id, bool is_init) {
    MS_LOG(WARNING) << "EnableDump ACL DataDumper";
  }
  virtual void Finalize() { MS_LOG(WARNING) << "Finalize ACL DataDumper"; }
};

class BACKEND_EXPORT DataDumperRegister {
 public:
  static DataDumperRegister &Instance();

  void RegistDumper(device::DeviceType backend, const std::shared_ptr<DataDumper> &dumper_ptr);

  std::shared_ptr<DataDumper> GetDumperForBackend(device::DeviceType backend);

 private:
  DataDumperRegister() = default;
  std::map<device::DeviceType, std::shared_ptr<DataDumper>> registered_dumpers_;
};

}  // namespace datadump
}  // namespace mindspore
#endif
