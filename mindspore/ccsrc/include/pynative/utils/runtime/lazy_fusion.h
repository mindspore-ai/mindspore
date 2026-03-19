/**
 * Copyright 2024-2026 Huawei Technologies Co., Ltd
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
#include <set>
#include <string>
#include <utility>
#include <functional>
#include "include/utils/visible.h"
#include "utils/ms_context.h"

namespace mindspore {
using LazyFusionInitFunc = std::function<bool()>;
using LazyFusionFlushFunc = std::function<void()>;

class PYNATIVE_UTILS_EXPORT LazyFusionFactory {
 public:
  LazyFusionFactory() = default;
  ~LazyFusionFactory() = default;

  void RegisterInit(const std::string &device_name, LazyFusionInitFunc &&func) { init_funcs_[device_name] = func; }
  void RegisterFlush(const std::string &device_name, LazyFusionFlushFunc &&func) { flush_funcs_[device_name] = func; }
  void Init();
  void Flush();

 private:
  std::map<std::string, LazyFusionInitFunc> init_funcs_;
  std::map<std::string, LazyFusionFlushFunc> flush_funcs_;
  std::set<std::string> enabled_devices_;
  friend void LazyFusionFlush();
};

extern PYNATIVE_UTILS_EXPORT LazyFusionFactory g_lazy_fusion;

class PYNATIVE_UTILS_EXPORT LazyFusionRegister {
 public:
  LazyFusionRegister(const std::string &device_name, LazyFusionInitFunc &&func) {
    g_lazy_fusion.RegisterInit(device_name, std::move(func));
  }
  ~LazyFusionRegister() = default;
};

class PYNATIVE_UTILS_EXPORT LazyFusionFlushRegister {
 public:
  LazyFusionFlushRegister(const std::string &device_name, LazyFusionFlushFunc &&func) {
    g_lazy_fusion.RegisterFlush(device_name, std::move(func));
  }
  ~LazyFusionFlushRegister() = default;
};

#define MS_REGISTER_LAZY_FUSION_INIT(DEVICE, FUNC) \
  static const LazyFusionRegister g_lazy_fusion_##DEVICE##_int_reg(DEVICE, FUNC)

#define MS_REGISTER_LAZY_FUSION_FLUSH(DEVICE, FUNC) \
  static const LazyFusionFlushRegister g_lazy_fusion_##DEVICE##_flush_reg(DEVICE, FUNC)

static inline void LazyFusionInit() { g_lazy_fusion.Init(); }

inline void LazyFusionFlush() {
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    return;
  }
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (g_lazy_fusion.enabled_devices_.find(device_name) == g_lazy_fusion.enabled_devices_.end()) {
    return;
  }
  g_lazy_fusion.Flush();
}
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_LAZY_FUSION_H_
