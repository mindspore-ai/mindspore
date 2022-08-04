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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_H_

#include <cstdint>
#include <string>
#include <utility>

namespace mindspore {
namespace parallel {
class Device {
  // This class abstract the 'device' information, used in Parallel module.
 public:
  Device() : rank_(0) { name_.clear(); }
  explicit Device(int64_t rank) : rank_(rank) { name_.clear(); }
  Device(std::string name, int64_t rank) : name_(std::move(name)), rank_(rank) {}
  ~Device() = default;
  std::string name() const { return name_; }
  int64_t rank() const { return rank_; }

 private:
  std::string name_;
  int64_t rank_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_H_
