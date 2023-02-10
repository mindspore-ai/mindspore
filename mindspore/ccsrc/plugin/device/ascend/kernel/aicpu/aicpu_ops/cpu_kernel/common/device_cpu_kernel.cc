/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel/common/device_cpu_kernel.h"

#include <string>

#include "aicpu_sharder/aicpu_context.h"
#include "cce/aicpu_engine_struct.h"
#include "cce/fwk_adpt_struct.h"
#include "cpu_kernel/common/cpu_kernel_cache.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "cpu_kernel/common/session_cache.h"
#include "cpu_kernel/common/status.h"

using namespace aicpu;
namespace {
// max param len limit 10k.
constexpr uint32_t kMaxParamLen = 10240;
// max extend info len limit 20k.
constexpr uint32_t kMaxExtendLen = 20480;
const std::string kContextKeyStreamId = "streamId";

uint32_t ParseExtSessionInfo(AicpuParamHead *param_head, SessionInfo *&session) {
  KERNEL_LOG_INFO("Parse extend session info begin.");
  uint32_t offset = 0;
  FWKAdapter::ExtInfo *ext_info = nullptr;
  char *ext_info_buf = reinterpret_cast<char *>(static_cast<uintptr_t>(param_head->extInfoAddr));
  while (offset + sizeof(FWKAdapter::ExtInfo) <= param_head->extInfoLength) {
    ext_info = reinterpret_cast<FWKAdapter::ExtInfo *>(ext_info_buf + offset);
    if (ext_info == nullptr) {
      KERNEL_LOG_ERROR(
        "Extend info is nullptr, extend info length[%u], extend info "
        "offset[%u].",
        param_head->extInfoLength, offset);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    if (ext_info->infoType == FWKAdapter::FWK_ADPT_EXT_SESSION_INFO) {
      auto need_len = sizeof(SessionInfo);
      if (ext_info->infoLen != need_len) {
        KERNEL_LOG_ERROR(
          "Parse extend session info failed, as info length must be "
          "[%zu], but %u.",
          sizeof(SessionInfo), ext_info->infoLen);
        return KERNEL_STATUS_PARAM_INVALID;
      }

      session = reinterpret_cast<SessionInfo *>(ext_info->infoMsg);
      KERNEL_LOG_INFO("Parse extend session info success.");
    }

    // not overflow
    offset += FWKAdapter::kExtInfoHeadSize;
    offset += ext_info->infoLen;
  }

  KERNEL_LOG_INFO("Parse extend session info end.");
  return KERNEL_STATUS_OK;
}
}  // namespace

extern "C" {
__attribute__((visibility("default"))) uint32_t RunCpuKernel(void *param) {
  KERNEL_LOG_INFO("RunCpuKernel C begin");
  if (param == nullptr) {
    KERNEL_LOG_ERROR("Param is null.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // parse param_len
  AicpuParamHead *param_head = static_cast<AicpuParamHead *>(param);
  if ((param_head->length < sizeof(AicpuParamHead)) || (param_head->length > kMaxParamLen) ||
      (param_head->extInfoLength > kMaxExtendLen)) {
    KERNEL_LOG_ERROR(
      "Param length[%u] not in [%zu, %u] or extend info length[%u] is "
      "greater "
      "than the limit[%u].",
      param_head->length, sizeof(AicpuParamHead), kMaxParamLen, param_head->extInfoLength, kMaxExtendLen);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  SessionInfo *session = nullptr;
  uint32_t ret = ParseExtSessionInfo(param_head, session);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  if (session == nullptr) {
    KERNEL_LOG_INFO("RunCpuKernel directly.");
    CpuKernelCache cache;
    cache.Init(false);
    return cache.RunKernel(param);
  }

  std::string stream_id_value = "0";
  auto status = GetThreadLocalCtx(kContextKeyStreamId, &stream_id_value);
  if (status != AICPU_ERROR_NONE) {
    KERNEL_LOG_ERROR("GetThreadLocalCtx failed, ret[%d].", status);
    return KERNEL_STATUS_INNER_ERROR;
  }
  uint64_t stream_id = atoi(stream_id_value.c_str());
  KERNEL_LOG_INFO(
    "RunCpuKernel from cache, stream id[%lu], session id[%lu], session "
    "flag[%d].",
    stream_id, session->sessionId, session->sessFlag);
  return SessionCache<CpuCacheData>::Instance().RunKernel<CpuKernelCache>(param, session->sessionId, stream_id,
                                                                          session->sessFlag);
}

__attribute__((visibility("default"))) uint32_t RunCpuKernelWithBlock(void *param, struct BlkDimInfo *blkdim_info) {
  KERNEL_LOG_INFO("RunCpuKernelWithBlock C begin. blockid[%u], blockdim[%u].", blkdim_info->blockId,
                  blkdim_info->blockNum);
  if (param == nullptr || blkdim_info == nullptr) {
    KERNEL_LOG_ERROR("Param is null.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // parse param_len
  AicpuParamHead *param_head = static_cast<AicpuParamHead *>(param);
  if ((param_head->length < sizeof(AicpuParamHead)) || (param_head->length > kMaxParamLen) ||
      (param_head->extInfoLength > kMaxExtendLen)) {
    KERNEL_LOG_ERROR(
      "Param length[%u] not in [%zu, %u] or extend info length[%u] is "
      "greater "
      "than the limit[%u].",
      param_head->length, sizeof(AicpuParamHead), kMaxParamLen, param_head->extInfoLength, kMaxExtendLen);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  SessionInfo *session = nullptr;
  uint32_t ret = ParseExtSessionInfo(param_head, session);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  if (session == nullptr) {
    KERNEL_LOG_INFO("RunCpuKernelWithBlock directly.");
    CpuKernelCache cache;
    cache.Init(false);
    return cache.RunCpuKernelWithBlock(param, blkdim_info);
  }

  std::string stream_id_value = "0";
  auto status = GetThreadLocalCtx(kContextKeyStreamId, &stream_id_value);
  if (status != AICPU_ERROR_NONE) {
    KERNEL_LOG_ERROR("GetThreadLocalCtx failed, ret[%d].", status);
    return KERNEL_STATUS_INNER_ERROR;
  }
  uint64_t stream_id = atoi(stream_id_value.c_str());
  KERNEL_LOG_INFO(
    "RunCpuKernel from cache, stream id[%lu], session id[%lu], session "
    "flag[%d].",
    stream_id, session->sessionId, session->sessFlag);
  return SessionCache<CpuCacheData>::Instance().RunCpuKernelWithBlock<CpuKernelCache>(
    param, session->sessionId, stream_id, session->sessFlag, blkdim_info);
}
}
