/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_COMMON_LOG_ADAPTER_H_
#define MINDSPORE_LITE_SRC_COMMON_LOG_ADAPTER_H_

#ifdef USE_GLOG
#include "utils/log_adapter.h"
#else
#include "src/common/log.h"
#endif  // USE_GLOG

namespace mindspore {
const char *const unsupport_string_tensor_log =
  "This mindspore-lite library does not support string tensors. Set environment variable MSLITE_ENABLE_STRING_KERNEL "
  "to on to "
  "recompile it.";
const char *const unsupport_controlflow_tensorlist_log =
  "This mindspore-lite library does not support controlflow and tensorlist op. Set environment variable "
  "MSLITE_ENABLE_CONTROLFLOW to on to recompile it.";
const char *const unsupport_auto_parallel_log =
  "The mindspore-lite library does not support auto parallel. Set environment variable MSLITE_ENABLE_AUTO_PARALLEL to "
  "on to "
  "recompile it.";
const char *const unsupport_weight_decode_log =
  "The mindspore-lite library does not support weight decode. Set environment variable MSLITE_ENABLE_WEIGHT_DECODE to "
  "on to "
  "recompile it.";
const char *const unsupport_custom_kernel_register_log =
  "The mindspore-lite library does not support custom kernel register. Set environment variable "
  "MSLITE_ENABLE_CUSTOM_KERNEL to on to "
  "recompile it.";
const char *const unsupport_delegate_log =
  "The mindspore-lite library does not support delegate. Set environment variable "
  "MSLITE_ENABLE_DELEGATE to on to "
  "recompile it.";
const char *const unsupport_fp16_log =
  "The mindspore-lite library does not support fp16. Set environment variable "
  "MSLITE_ENABLE_FP16 to on to "
  "recompile it.";
const char *const unsupport_int8_log =
  "The mindspore-lite library does not support int8. Set environment variable "
  "MSLITE_ENABLE_INT8 to on to "
  "recompile it.";

static inline bool IsPrintDebug() {
  auto env = std::getenv("GLOG_v");
  return env != nullptr && env[0] == '0';
}
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_LOG_ADAPTER_H_
