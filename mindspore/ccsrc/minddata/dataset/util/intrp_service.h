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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_INTRP_SERVICE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_INTRP_SERVICE_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/intrp_resource.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/service.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
using SvcAllocator = Allocator<std::pair<const std::string, IntrpResource *>>;

class IntrpService : public Service {
 public:
  IntrpService();

  ~IntrpService() noexcept override;

  IntrpService(const IntrpService &) = delete;

  IntrpService &operator=(const IntrpService &) = delete;

  Status Register(const std::string &name, IntrpResource *res);

  Status Deregister(const std::string &name) noexcept;

  void InterruptAll() noexcept;

  Status DoServiceStart() override { return Status::OK(); }

  Status DoServiceStop() override { return Status::OK(); }

 private:
  int high_water_mark_;
  std::mutex mutex_;
  std::map<std::string, IntrpResource *> all_intrp_resources_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_INTRP_SERVICE_H_
