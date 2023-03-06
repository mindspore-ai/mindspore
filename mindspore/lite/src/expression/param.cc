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

#include "src/expression/param.h"
#include <random>
#include <algorithm>
#include <string>
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

constexpr float kZero = 0.0f;
constexpr float kOne = 1.0f;

namespace mindspore {
namespace lite {
int Param::Fill(Mode mode) {
  std::default_random_engine engine{static_cast<unsigned int>(0)};
  std::vector<float> data(size_);
  switch (mode) {
    case NORMAL: {
      constexpr float scale = 0.01;
      std::normal_distribution<float> n{0, 1};
      std::generate_n(data.begin(), size_, [&]() { return n(engine); });
      (void)std::transform(data.begin(), data.end(), data.begin(), [=](float x) { return x * scale; });
      break;
    }
    case UNIFORM: {
      constexpr float scale = 0.07;
      std::uniform_real_distribution<float> u{-1.0, 1.0};
      std::generate_n(data.begin(), size_, [&]() { return u(engine) * scale; });
      break;
    }
    case ZEROS:
      std::fill_n(data.begin(), size_, kZero);
      break;
    case ONES:
      std::fill_n(data.begin(), size_, kOne);
      break;
    case NOT_SUPPORTED:
      return RET_ERROR;
  }
  Copy(data);
  return RET_OK;
}

Param::Mode Param::String2Enum(std::string mode) {
  (void)std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
  if (mode == "normal") return NORMAL;
  if (mode == "uniform") return UNIFORM;
  if (mode == "ones") return ONES;
  if (mode == "zeors") return ZEROS;
  return NOT_SUPPORTED;
}
}  // namespace lite
}  // namespace mindspore
