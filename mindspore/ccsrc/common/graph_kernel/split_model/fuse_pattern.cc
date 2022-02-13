/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "common/graph_kernel/split_model/fuse_pattern.h"
#include <sstream>

namespace mindspore::graphkernel::inner {
std::string FusePattern::ToString() const {
  std::ostringstream oss;
  if (direction_ == FuseDirection::FORWARD) {
    oss << "Forward{";
  } else {
    oss << "Backward{";
  }
  bool first = true;
  for (auto &area : fused_areas_) {
    if (first) {
      first = false;
    } else {
      oss << ",";
    }
    oss << area->ToString();
  }
  oss << "}";
  return oss.str();
}

bool FuseVirtualNode::Match(const AreaPtr &area) {
  fused_areas_ = area->inputs();
  return true;
}
}  // namespace mindspore::graphkernel::inner
