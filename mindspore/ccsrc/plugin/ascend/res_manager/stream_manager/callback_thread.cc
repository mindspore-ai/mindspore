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

#include "plugin/ascend/res_manager/stream_manager/callback_thread.h"
#include "utils/log_adapter.h"
#include "acl/error_codes/rt_error_codes.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
static const int32_t kProcessReportTimeout = 500;

void *callback_thread_func(void *data) {
  pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, nullptr);
#ifdef WITH_BACKEND
  auto flag_ptr = static_cast<std::shared_ptr<std::atomic_bool> *>(data);
  // Copy the shared_ptr to increase reference count
  auto flag_shared = *flag_ptr;
  while (flag_shared->load()) {
    try {
      auto ret = CALL_ASCEND_API(aclrtProcessReport, kProcessReportTimeout);
      if (ret && ret != ACL_ERROR_WAIT_CALLBACK_TIMEOUT && ret != ACL_ERROR_RT_REPORT_TIMEOUT) {
        MS_LOG(DEBUG) << "aclrtProcessReport err : " << ret << ".";
      }
    } catch (const std::exception &ex) {
      MS_LOG(ERROR) << "aclrtProcessReport exception : " << ex.what() << ".";
      break;
    }
  }
  MS_LOG(INFO) << "Exit callback thread loop.";
#endif
  return data;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
