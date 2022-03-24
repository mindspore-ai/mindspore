/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_STUB_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_STUB_H_

#include <memory>
#include <string>
#include "minddata/dataset/engine/cache/cache_common.h"
#include "minddata/dataset/engine/cache/cache_request.h"
#include "minddata/dataset/util/service.h"

namespace mindspore {
namespace dataset {
class CacheClientGreeter : public Service {
 public:
  explicit CacheClientGreeter(const std::string &hostname, int32_t port, int32_t num_workers) {}
  ~CacheClientGreeter() override {}
  Status DoServiceStart() override { RETURN_STATUS_UNEXPECTED("Not supported"); }
  Status DoServiceStop() override { RETURN_STATUS_UNEXPECTED("Not supported"); }

  void *SharedMemoryBaseAddr() { return nullptr; }
  Status HandleRequest(std::shared_ptr<BaseRequest> rq) { RETURN_STATUS_UNEXPECTED("Not supported"); }
  Status AttachToSharedMemory(bool *local_bypass) { RETURN_STATUS_UNEXPECTED("Not supported"); }
  std::string GetHostname() const { return "Not supported"; }
  int32_t GetPort() const { return 0; }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_STUB_H_
