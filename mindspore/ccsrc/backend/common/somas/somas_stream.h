/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_STREAM_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_STREAM_H_

#include "backend/common/somas/somas_node.h"

#include <memory>
#include <set>
#include <vector>

namespace mindspore {
namespace somas {
class SomasStream {
 public:
  // Attributes mutated in code
  std::vector<SomasNodePtr> nodes_;

  // Constructors/Destructors
  explicit SomasStream(size_t id) : id_(id) {}
  SomasStream(const SomasStream &) = delete;
  SomasStream &operator=(const SomasStream &) = delete;
  ~SomasStream() = default;

  // Accessors
  const size_t &GetId() const { return id_; }

 private:
  const size_t id_{0};
};
using SomasStreamPtr = std::shared_ptr<SomasStream>;
}  // namespace somas
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_STREAM_H_
