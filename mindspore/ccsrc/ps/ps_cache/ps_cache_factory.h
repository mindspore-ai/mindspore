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
#ifndef MINDSPORE_CCSRC_PS_PS_CACHE_PS_CACHE_FACTORY_H_
#define MINDSPORE_CCSRC_PS_PS_CACHE_PS_CACHE_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "ps/ps_cache/ps_cache_basic.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace ps {
using PsCacheCreator = std::function<std::shared_ptr<PsCacheBasic>()>;
class PsCacheFactory {
 public:
  static PsCacheFactory &Get();
  void Register(const std::string &device_name, PsCacheCreator &&ps_cache_creator);
  std::shared_ptr<PsCacheBasic> ps_cache(const std::string &device_name);

 private:
  PsCacheFactory() = default;
  ~PsCacheFactory() = default;
  DISABLE_COPY_AND_ASSIGN(PsCacheFactory)
  std::map<std::string, PsCacheCreator> ps_cache_creators_;
};

class PsCacheRegistrar {
 public:
  PsCacheRegistrar(const std::string &device_name, PsCacheCreator &&ps_cache_creator) {
    PsCacheFactory::Get().Register(device_name, std::move(ps_cache_creator));
  }
  ~PsCacheRegistrar() = default;
};

#define MS_REG_PS_CACHE(DEVICE_NAME, PS_CACHE_CLASS)                          \
  static const PsCacheRegistrar g_ps_cache_registrar__##DEVICE_NAME##_##_reg( \
    DEVICE_NAME, []() { return std::make_shared<PS_CACHE_CLASS>(); });
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_PS_CACHE_PS_CACHE_FACTORY_H_
