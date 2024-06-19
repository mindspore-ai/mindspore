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

#ifndef MINDSPORE_CCSRC_PLUGIN_MINDIO_MINDIOADAPTER_H
#define MINDSPORE_CCSRC_PLUGIN_MINDIO_MINDIOADAPTER_H

#include <thread>
#include <functional>
#include <memory>
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace mindio {

ORIGIN_METHOD(TTP_NotifyStartUpdatingOs, int, int64_t);
ORIGIN_METHOD(TTP_NotifyEndUpdatingOs, int, int64_t);

class MindIOAdapter {
 public:
  MindIOAdapter() = default;
  virtual ~MindIOAdapter() = default;
  MindIOAdapter(const MindIOAdapter &) = delete;
  MindIOAdapter &operator=(const MindIOAdapter &) = delete;

  void SetOsStateNotifyCallBack(const TTP_NotifyStartUpdatingOsFunObj &opStart,
                                const TTP_NotifyEndUpdatingOsFunObj &optEnd);
  void NotifyStartUpdatingOs();
  void NotifyEndUpdatingOs();
  bool IsEnable();
  static std::shared_ptr<MindIOAdapter> GetInstance();
  static std::shared_ptr<MindIOAdapter> inst_mindio_;

 private:
  TTP_NotifyStartUpdatingOsFunObj _optStart = nullptr;
  TTP_NotifyEndUpdatingOsFunObj _optEnd = nullptr;
  bool isEnable;
};

}  // namespace mindio
}  // namespace mindspore
#endif
