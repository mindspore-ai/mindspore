/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_MODEL_CONTEXT_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_MODEL_CONTEXT_H_

#include <vector>
#include "runtime/rt_model.h"

namespace mindspore::ge::model_runner {
class ModelContext {
 public:
  ModelContext(uint32_t device_id, uint64_t session_id, int32_t priority, rtModel_t rt_model_handle,
               rtStream_t rt_model_stream, const std::vector<rtStream_t> &stream_list,
               const std::vector<rtLabel_t> &label_list, const std::vector<rtEvent_t> &event_list)
      : device_id_(device_id),
        session_id_(session_id),
        priority_(priority),
        rt_model_handle_(rt_model_handle),
        rt_model_stream_(rt_model_stream),
        stream_list_(stream_list),
        label_list_(label_list),
        event_list_(event_list) {}
  ~ModelContext() {}

  uint64_t device_id() const { return device_id_; }
  uint64_t session_id() const { return session_id_; }
  int32_t priority() const { return priority_; }
  const rtModel_t &rt_model_handle() const { return rt_model_handle_; }
  const rtStream_t &rt_model_stream() const { return rt_model_stream_; }
  const std::vector<rtStream_t> &stream_list() const { return stream_list_; }
  const std::vector<rtLabel_t> &label_list() const { return label_list_; }
  const std::vector<rtEvent_t> &event_list() const { return event_list_; }

 private:
  uint32_t device_id_;
  uint64_t session_id_;
  int32_t priority_;
  rtModel_t rt_model_handle_;
  rtStream_t rt_model_stream_;
  std::vector<rtStream_t> stream_list_;
  std::vector<rtLabel_t> label_list_;
  std::vector<rtEvent_t> event_list_;
};
}  // namespace mindspore::ge::model_runner
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_MODEL_CONTEXT_H_
