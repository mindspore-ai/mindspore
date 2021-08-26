/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/device/ascend/distribute/ascend_collective.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace collective {
HcclCollectiveGroup &HcclCollectiveGroup::instance() {
  static HcclCollectiveGroup instance;
  return instance;
}
int HcclCollectiveGroup::GetRankSize(const std::string &) const { return 0; }
int HcclCollectiveGroup::GetRankId(const std::string &) const { return 0; }
int HcclCollectiveGroup::GetDeviceId() const { return 0; }
void HcclCollectiveGroup::CreateCommGroup(const std::string &, const std::vector<unsigned int> &) { return; }
void HcclCollectiveGroup::FinalizeCollective() { return; }
}  // namespace collective
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
