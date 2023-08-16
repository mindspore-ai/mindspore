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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_CONTEXT_MANAGER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_CONTEXT_MANAGER_H_

#include <string>
#include <memory>
#include <map>
#include "runtime/context.h"
#include "runtime/stream.h"

namespace mindspore {
class GeContextManager {
 public:
  GeContextManager();
  ~GeContextManager();

  GeContextManager(const GeContextManager &) = delete;
  GeContextManager &operator=(const GeContextManager &) = delete;

  bool InitContext(uint32_t device_id);
  bool SetContext();
  void DestroyContext();
  rtStream_t GetDefaultStream();
  bool SyncStream(rtStream_t stream) const;

 private:
  uint32_t device_id_ = 0;
  rtContext_t context_ = nullptr;
  rtStream_t default_stream_ = nullptr;
  void DestroyDefaultStream();
  bool CreateDefaultStream();
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_CONTEXT_MANAGER_H_
