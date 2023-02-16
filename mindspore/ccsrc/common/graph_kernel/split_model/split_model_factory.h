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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_SPLIT_MODEL_FACTORY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_SPLIT_MODEL_FACTORY_H_

#include <memory>
#include <string>
#include "common/graph_kernel/split_model/split_model.h"
#include "utils/hash_map.h"
#include "include/backend/visible.h"

namespace mindspore::graphkernel::inner {
class BACKEND_EXPORT SplitModelFactory {
 public:
  static SplitModelFactory &Instance() {
    static SplitModelFactory instance = SplitModelFactory();
    return instance;
  }
  SplitModelPtr CreateSplitModel(const std::string &processor);
  using RegFunc = std::function<std::shared_ptr<SplitModel>()>;
  void Register(const std::string &processor, const RegFunc &func) { creators[processor] = func; }

 private:
  mindspore::HashMap<std::string, RegFunc> creators;
};

class SplitModelRegister {
 public:
  SplitModelRegister(const std::string &processor, const SplitModelFactory::RegFunc &func) : func_(func) {
    SplitModelFactory::Instance().Register(processor, func);
  }
  ~SplitModelRegister() = default;

 protected:
  // for pclint-plus
  SplitModelFactory::RegFunc func_;
};

#define SPLIT_MODEL_REGISTER(processor, cls) \
  const SplitModelRegister split_model(      \
    processor, []() noexcept { return std::static_pointer_cast<SplitModel>(std::make_shared<cls>()); })
}  // namespace mindspore::graphkernel::inner
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_SPLIT_MODEL_FACTORY_H_
