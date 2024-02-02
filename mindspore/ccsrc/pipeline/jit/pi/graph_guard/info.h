/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_PI_JIT_INFO_H
#define MINDSPORE_PI_JIT_INFO_H

#include <string>
#include <memory>
#include <vector>

namespace mindspore {
namespace pijit {

constexpr size_t INVALID_HASH = size_t(-1);
constexpr size_t INVALID_ID = size_t(-1);

class InfoPack {
 public:
  InfoPack();
  InfoPack(const InfoPack &);
  size_t Hash() const;
  size_t Id() const;
  std::string ToString() const;
  void Update();
  bool operator()(const InfoPack &lhs, const InfoPack &rhs) const;
  size_t operator()(const InfoPack &k) const;
  InfoPack &Begin();
  InfoPack &End();
  InfoPack &operator<<(int8_t v);
  InfoPack &operator<<(uint8_t v);
  InfoPack &operator<<(int16_t v);
  InfoPack &operator<<(uint16_t v);
  InfoPack &operator<<(int32_t v);
  InfoPack &operator<<(uint32_t v);
  InfoPack &operator<<(int64_t v);
  InfoPack &operator<<(uint64_t v);
  InfoPack &operator<<(float v);
  InfoPack &operator<<(double v);
  InfoPack &operator<<(bool v);
  InfoPack &operator<<(void *v);
  InfoPack &operator<<(const std::string &v);
  InfoPack &operator<<(const std::vector<int8_t> &v);
  InfoPack &operator<<(const std::vector<uint8_t> &v);
  InfoPack &operator<<(const std::vector<int16_t> &v);
  InfoPack &operator<<(const std::vector<uint16_t> &v);
  InfoPack &operator<<(const std::vector<int32_t> &v);
  InfoPack &operator<<(const std::vector<uint32_t> &v);
  InfoPack &operator<<(const std::vector<int64_t> &v);
  InfoPack &operator<<(const std::vector<uint64_t> &v);
  InfoPack &operator<<(const std::vector<float> &v);
  InfoPack &operator<<(const std::vector<double> &v);
  InfoPack &operator<<(const std::vector<bool> &v);
  InfoPack &operator<<(const std::vector<std::string> &v);
  InfoPack &operator<<(const std::vector<void *> &v);
  InfoPack &operator<<(const InfoPack &v);

 protected:
  size_t CalcHash(std::string v);
  size_t CalcId(std::string v);
  size_t CalcString(std::string v);
  size_t hash_;
  size_t id_;
  std::string info_;
};
using InfoPackPtr = std::shared_ptr<InfoPack>;

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_INFO_H
