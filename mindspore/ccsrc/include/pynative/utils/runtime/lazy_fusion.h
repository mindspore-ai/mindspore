/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_LAZY_FUSION_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_LAZY_FUSION_H_

#include <map>
#include <string>
#include <utility>
#include <functional>
#include "include/utils/visible.h"

namespace mindspore {
using LazyFusionInitFunc = std::function<void()>;

class PYNATIVE_UTILS_EXPORT LazyFusionFactory {
 public:
  LazyFusionFactory() = default;
  ~LazyFusionFactory() = default;

  void Register(const std::string &device_name, LazyFusionInitFunc &&func) { funcs_[device_name] = func; }
  void Init();

 private:
  std::map<std::string, LazyFusionInitFunc> funcs_;
};

extern PYNATIVE_UTILS_EXPORT LazyFusionFactory g_lazy_fusion;

class PYNATIVE_UTILS_EXPORT LazyFusionRegister {
 public:
  LazyFusionRegister(const std::string &device_name, LazyFusionInitFunc &&func) {
    g_lazy_fusion.Register(device_name, std::move(func));
  }
  ~LazyFusionRegister() = default;
};

#define MS_REGISTER_LAZY_FUSION_INIT(DEVICE, FUNC) \
  static const LazyFusionRegister g_lazy_fusion_##DEVICE##_int_reg(DEVICE, FUNC)

static inline void LazyFusionInit() { g_lazy_fusion.Init(); }
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_LAZY_FUSION_H_
