/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_FACTORY_MGR_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_FACTORY_MGR_BASE_H_
#include <map>
#include <memory>
#include "include/api/status.h"

namespace mindspore {
namespace lite {
template <typename KEY, typename PRODUCT>
class ProcductRegistrar {
 public:
  virtual std::shared_ptr<PRODUCT> Create() = 0;

 protected:
  ProcductRegistrar() {}
  virtual ~ProcductRegistrar() {}

 private:
  ProcductRegistrar(const ProcductRegistrar &);
  const ProcductRegistrar &operator=(const ProcductRegistrar &);
};

template <typename KEY, typename PRODUCT>
class FactoryManagerBase {
 public:
  static FactoryManagerBase &Instance() {
    static FactoryManagerBase<KEY, PRODUCT> instance;
    return instance;
  }
  void RegProduct(const KEY &key, ProcductRegistrar<KEY, PRODUCT> *registrar) { registrars[key] = registrar; }

  std::shared_ptr<PRODUCT> GetProduct(const KEY &key) {
    auto registrar_iter = registrars.find(key);
    if (registrar_iter != registrars.end()) {
      if (registrar_iter->second != nullptr) {
        return registrar_iter->second->Create();
      }
    }
    return nullptr;
  }

 private:
  FactoryManagerBase() = default;
  ~FactoryManagerBase() = default;
  FactoryManagerBase(const FactoryManagerBase &);
  const FactoryManagerBase &operator=(const FactoryManagerBase &);

 private:
  std::map<KEY, ProcductRegistrar<KEY, PRODUCT> *> registrars;
};

template <typename KEY, typename PRODUCT, typename PRODUCT_IMPL>
class CommonProcductRegistrar : public ProcductRegistrar<KEY, PRODUCT> {
 public:
  explicit CommonProcductRegistrar(const KEY &key) {
    FactoryManagerBase<KEY, PRODUCT>::Instance().RegProduct(key, this);
  }
  std::shared_ptr<PRODUCT> Create() { return std::make_shared<PRODUCT_IMPL>(); }
};

#define RET_COMMON_PRODUCT_REGISTRAR(KEY, PRODUCT, PRODUCT_IMPL, key, name) \
  static mindspore::lite::CommonProcductRegistrar<KEY, PRODUCT, PRODUCT_IMPL> g_commonProcductRegistrar##name(key);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_FACTORY_MGR_BASE_H_
