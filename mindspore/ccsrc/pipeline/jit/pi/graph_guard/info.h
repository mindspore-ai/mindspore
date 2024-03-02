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
#include "pybind11/pybind11.h"
#include "mindspore/core/base/base.h"

namespace py = pybind11;

namespace mindspore {
namespace pijit {

constexpr size_t kInvalidId = size_t(-1);

class InfoPack {
 public:
  InfoPack();
  InfoPack(const InfoPack &);
  virtual ~InfoPack();
  size_t Id() const;
  uint8_t *Buf(size_t *sz) const;
  void Update();
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
  InfoPack &operator<<(PyObject *v);
  InfoPack &operator<<(mindspore::BasePtr v);
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
  InfoPack &operator<<(const std::vector<PyObject *> &v);
  InfoPack &operator<<(const InfoPack &v);

 protected:
  size_t CalcBuffer(uint8_t *buf, size_t sz);
  size_t CalcString(std::string v);
  void AllocIfNeed(size_t need);
  size_t id_;
  std::unique_ptr<uint8_t[]> buf_;
  size_t ptr_;
  size_t limit_;
};
using InfoPackPtr = std::shared_ptr<InfoPack>;

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_INFO_H
