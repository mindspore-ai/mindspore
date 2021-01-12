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

#ifndef MINDSPORE_CCSRC_UTILS_CONTEXT_CONTEXT_EXTENDS_H
#define MINDSPORE_CCSRC_UTILS_CONTEXT_CONTEXT_EXTENDS_H

#include <map>
#include <string>
#include <memory>
#include "utils/ms_context.h"
#include "utils/tensorprint_utils.h"

#ifndef NO_DLIB
#include "acl/acl_tdt.h"
#include "tdt/tsd_client.h"
#include "tdt/data_common.h"
#include "runtime/dev.h"
#endif
#ifdef ENABLE_GE
#include "transform/graph_ir/df_graph_manager.h"
#endif

namespace mindspore {
namespace context {
bool OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr);
bool CloseTsd(const std::shared_ptr<MsContext> &ms_context_ptr, bool force = false);
void SetHcclOptions(const std::shared_ptr<MsContext> &inst_context, std::map<std::string, std::string> *ge_options);
void GetGeOptions(const std::shared_ptr<MsContext> &inst_context, std::map<std::string, std::string> *ge_options);
void SetDisableReuseMemoryFlag(std::map<std::string, std::string> *ge_options);
bool InitGe(const std::shared_ptr<MsContext> &inst_context);
bool FinalizeGe(const std::shared_ptr<MsContext> &inst_context, bool force = false);
bool PynativeInitGe(const std::shared_ptr<MsContext> &inst_context);
bool IsTsdOpened(const std::shared_ptr<MsContext> &inst_context);
bool IsGeInited(const std::shared_ptr<MsContext> &inst_context);
}  // namespace context
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_CONTEXT_CONTEXT_EXTENDS_H
