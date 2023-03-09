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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_PASS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_PASS_H_
#include <memory>
#include <string>

#include "ir/anf.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
class CacheManager;
using CacheManagerPtr = std::shared_ptr<CacheManager>;

// @brief ANF Graph level optimization base pass
class Pass {
 public:
  explicit Pass(const std::string &name = "pass") : name_(name) {}
  virtual ~Pass() = default;
  virtual bool Run(const FuncGraphPtr &func_graph) = 0;
  const std::string &name() const { return name_; }
  void SetCacheManager(const CacheManagerPtr &cm) { cache_manager_ = cm; }
  const CacheManagerPtr &GetCacheManager() const { return cache_manager_; }

 private:
  const std::string name_;
  CacheManagerPtr cache_manager_;
};
using PassPtr = std::shared_ptr<Pass>;
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_PASS_H_
