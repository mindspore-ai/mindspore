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

#ifndef MINDSPORE_ASCEND_CAPTURE_GRAPH_H
#define MINDSPORE_ASCEND_CAPTURE_GRAPH_H

#include "acl/acl_mdl.h"
#include "backend/ms_backend/runtime/graph_executor/kernel_capture/capture_graph.h"

namespace mindspore::device::ascend {

class AscendCaptureGraph : public CaptureGraph {
 public:
  AscendCaptureGraph() = default;
  ~AscendCaptureGraph() override;
  bool CaptureBegin(uint32_t stream_id) override;
  void CaptureGetInfo(uint32_t stream_id) override;
  void CaptureEnd(uint32_t stream_id) override;
  void ExecuteCaptureGraph(uint32_t stream_id) override;

 protected:
  aclrtStream capture_stream_{nullptr};
#if defined(__linux__) && defined(WITH_BACKEND)
  aclmdlRICaptureMode mode_{aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED};
  aclmdlRI model_ri_{nullptr};
#endif
  bool finish_capture_graph_{false};
};
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_ASCEND_CAPTURE_GRAPH_H
