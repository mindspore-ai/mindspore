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

#include "plugin/ascend/res_manager/capture_graph/ascend_capture_graph.h"

#include <cstdint>
#include <string>

#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_mdl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/event/ascend_event.h"

#include "utils/log_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"

namespace mindspore::device::ascend {
AscendCaptureGraph::~AscendCaptureGraph() {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (finish_capture_graph_ && model_ri_) {
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
      MS_LOG(ERROR) << "Sync All streams failed";
    }
    auto ret = CALL_ASCEND_API(aclmdlRIDestroy, model_ri_);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(WARNING) << "aclmdlRIDestroy failed, ret:" << ret;
    }
  }
#endif
}

bool AscendCaptureGraph::CaptureBegin(uint32_t stream_id) {
  if (finish_capture_graph_) {
    MS_LOG(ERROR) << "Already capture a graph.";
    return false;
  }

  capture_stream_ = AscendStreamMng::GetInstance().GetStream(stream_id);
#if defined(__linux__) && defined(WITH_BACKEND)
  auto ret = CALL_ASCEND_API(aclmdlRICaptureBegin, capture_stream_, mode_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclmdlRICaptureBegin failed, ret:" << ret;
    return false;
  }
#endif
  return true;
}

void AscendCaptureGraph::CaptureGetInfo(uint32_t stream_id) {
  auto current_stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(current_stream);
  MS_EXCEPTION_IF_NULL(capture_stream_);
  if (current_stream != capture_stream_) {
    MS_LOG(EXCEPTION) << "The current stream is not in capture status.";
  }
#if defined(__linux__) && defined(WITH_BACKEND)
  aclmdlRICaptureStatus status;
  auto ret = CALL_ASCEND_API(aclmdlRICaptureGetInfo, capture_stream_, &status, &model_ri_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "aclmdlRICaptureGetInfo failed, ret:" << ret;
  }
  if (status != aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
    MS_LOG(EXCEPTION) << "aclmdlRICaptureGetInfo got wrong status: " << status;
  }
#endif
}

void AscendCaptureGraph::CaptureEnd(uint32_t stream_id) {
  auto current_stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(current_stream);
  MS_EXCEPTION_IF_NULL(capture_stream_);
  if (current_stream != capture_stream_) {
    MS_LOG(EXCEPTION) << "The current stream is not in capture status.";
  }
#if defined(__linux__) && defined(WITH_BACKEND)
  auto ret = CALL_ASCEND_API(aclmdlRICaptureEnd, capture_stream_, &model_ri_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "aclmdlRICaptureEnd failed, ret:" << ret;
  }

  finish_capture_graph_ = true;
#endif
}

void AscendCaptureGraph::ExecuteCaptureGraph(uint32_t stream_id) {
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream);
#if defined(__linux__) && defined(WITH_BACKEND)
  MS_EXCEPTION_IF_NULL(model_ri_);

  auto ret = CALL_ASCEND_API(aclmdlRIExecuteAsync, model_ri_, stream);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "aclmdlRIExecuteAsync failed, ret:" << ret;
  }
#endif
}
}  // namespace mindspore::device::ascend
