/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <thread>
#include <atomic>
#include <memory>
#include <mutex>
#include <algorithm>
#include <utility>
#include "utils/log_adapter.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "mindio/mindio_adapter.h"

namespace mindspore {
namespace mindio {

std::shared_ptr<MindIOAdapter> MindIOAdapter::inst_mindio_ = nullptr;

std::shared_ptr<MindIOAdapter> MindIOAdapter::GetInstance() {
  static std::once_flag inst_mindio_init_flag_ = {};
  std::call_once(inst_mindio_init_flag_, [&]() {
    MS_LOG(INFO) << "Start create new mindio adapter instance.";
    if (inst_mindio_ == nullptr) {
      inst_mindio_ = std::make_shared<MindIOAdapter>();
      auto env = common::GetEnv("MS_ENABLE_MINDIO_GRACEFUL_EXIT");
      auto context = MsContext::GetInstance();
      bool isEnable = true;
      int execute_mode = context->get_param<int>(MS_CTX_EXECUTION_MODE);
      if (env.empty() || execute_mode != kGraphMode) {
        isEnable = false;
      }
      if (isEnable) {
        isEnable = false;
        void *handle = dlopen("/usr/local/mindio/libttp.so", RTLD_LAZY);
        auto libPath = common::GetEnv("MS_MINDIO_TTP_LIB_PATH");
        if (handle == nullptr && (!libPath.empty())) {
          MS_LOG(INFO) << "Default so path is incorrect and found custom mindio so path";
          handle = dlopen(libPath.c_str(), RTLD_LAZY);
        }
        if (handle) {
          MS_LOG(INFO) << "Found mindio so.";
          auto startFunc = DlsymWithCast<TTP_NotifyStartUpdatingOsFunPtr>(handle, "MindioTtpSetOptimStatusUpdating");
          auto endFunc = DlsymWithCast<TTP_NotifyEndUpdatingOsFunPtr>(handle, "MindioTtpSetOptimStatusFinished");
          if (startFunc && endFunc) {
            MS_LOG(INFO) << "Found mindio symbols.";
            inst_mindio_->SetOsStateNotifyCallBack(startFunc, endFunc);
            isEnable = true;
          }
        }
      }
      inst_mindio_->isEnable = isEnable;
      MS_LOG(INFO) << "Finish create new mindio adapter instance, isEnable:" << isEnable;
    }
  });
  MS_EXCEPTION_IF_NULL(inst_mindio_);
  return inst_mindio_;
}

bool MindIOAdapter::IsEnable() { return isEnable; }

void MindIOAdapter::SetOsStateNotifyCallBack(const TTP_NotifyStartUpdatingOsFunObj &optStart,
                                             const TTP_NotifyEndUpdatingOsFunObj &optEnd) {
  _optStart = optStart;
  _optEnd = optEnd;
}
void MindIOAdapter::NotifyStartUpdatingOs() {
  if (_optStart != nullptr) {
    auto ret = _optStart(-1);
    MS_LOG(INFO) << "Notify start updating optimizer event to mindio. ret=" << ret;
  }
}
void MindIOAdapter::NotifyEndUpdatingOs() {
  if (_optEnd != nullptr) {
    auto ret = _optEnd(-1);
    MS_LOG(INFO) << "Notify Finish updating optimizer event to mindio. ret=" << ret;
  }
}
}  // namespace mindio
}  // namespace mindspore
