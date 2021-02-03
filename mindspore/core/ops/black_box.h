/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_BLACK_BOX_H_
#define MINDSPORE_CORE_OPS_BLACK_BOX_H_
#include <string>
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBlackBox = "BlackBox";
class BlackBox : public PrimitiveC {
 public:
  BlackBox() : PrimitiveC(kNameBlackBox) {}
  ~BlackBox() = default;
  MS_DECLARE_PARENT(BlackBox, PrimitiveC);
  void Init(const std::string &id, const int64_t size, const std::vector<int64_t> &address);
  void set_id(const std::string &id);
  void set_size(const int64_t size);
  void set_address(const std::vector<int64_t> &address);
  std::string get_id() const;
  int64_t get_size() const;
  std::vector<int64_t> get_address() const;
};

using PrimBlackBoxPtr = std::shared_ptr<BlackBox>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BLACK_BOX_H_
