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
#ifndef MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_CHECKSUM_CHECKSUM_MGR_H_
#define MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_CHECKSUM_CHECKSUM_MGR_H_

#include <shared_mutex>
#include "tools/visible.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace checksum {
class TOOLS_EXPORT CheckSumMgr {
 public:
  static CheckSumMgr &GetInstance() {
    static CheckSumMgr instance;
    return instance;
  }
  ~CheckSumMgr() = default;
  bool NeedEnableCheckSum() const;
  bool IsCheckSumEnable() const;
  void CheckSumStart();
  void CheckSumStop();
  bool GetCheckSumResult() const;
  void SetCheckSumResult(bool result);

 private:
  CheckSumMgr() = default;
  DISABLE_COPY_AND_ASSIGN(CheckSumMgr);
  bool enable_{false};
  bool result_{false};
  mutable std::shared_mutex enable_mutex_;
  mutable std::shared_mutex result_mutex_;
};

bool NeedEnableCheckSum();

}  // namespace checksum
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_CHECKSUM_CHECKSUM_MGR_H_
