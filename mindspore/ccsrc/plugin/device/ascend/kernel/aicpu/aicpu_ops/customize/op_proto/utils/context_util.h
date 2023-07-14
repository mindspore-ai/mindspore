/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

/*!
 * \file context_util.h
 * \brief
 */

#ifndef CANN_CUSTOMIZE_CONTEXT_UTIL_H_
#define CANN_CUSTOMIZE_CONTEXT_UTIL_H_

#include "runtime/infer_shape_context.h"
#include "runtime/tiling_context.h"
#include "op_log.h"

namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr);                                        \
    REPORT_CALL_ERROR("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                           \
    return ge::GRAPH_FAILED;                                                                     \
  }

#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                       \
  if ((ptr) == nullptr) {                                                                        \
    const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr);                                        \
    REPORT_CALL_ERROR("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                           \
    return ret;                                                                                  \
  }
}  // namespace ops
#endif  // CANN_CUSTOMIZE_CONTEXT_UTIL_H_
