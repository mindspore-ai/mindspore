/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_CAPTURE_GRAPH_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_CAPTURE_GRAPH_H_

#include <memory>
#include <vector>

namespace mindspore {
class CaptureGraph {
 public:
  virtual ~CaptureGraph() = default;
  virtual bool CaptureBegin(uint32_t stream_id) = 0;
  virtual void CaptureGetInfo(uint32_t stream_id) = 0;
  virtual void CaptureEnd(uint32_t stream_id) = 0;
  virtual void ExecuteCaptureGraph(uint32_t stream_id) = 0;
};
using CaptureGraphPtr = std::shared_ptr<CaptureGraph>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_CAPTURE_GRAPH_H_
