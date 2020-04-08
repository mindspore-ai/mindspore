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
#include "dataset/core/global_context.h"

#include <memory>
#include <mutex>

#include "dataset/core/config_manager.h"
#include "dataset/core/cv_tensor.h"
#include "dataset/core/tensor.h"
#include "dataset/util/allocator.h"
#include "dataset/util/circular_pool.h"
#include "dataset/util/system_pool.h"

namespace mindspore {
namespace dataset {
// Global static pointer for the singleton GlobalContext
std::unique_ptr<GlobalContext> GlobalContext::global_context_ = nullptr;
std::once_flag GlobalContext::init_instance_flag_;

constexpr int GlobalContext::kArenaSize;
constexpr int GlobalContext::kMaxSize;
constexpr bool GlobalContext::kInitArena;

// Singleton initializer
GlobalContext *GlobalContext::Instance() {
  // If the single global context is not created yet, then create it. Otherwise the
  // existing one is returned.
  std::call_once(init_instance_flag_, []() {
    global_context_.reset(new GlobalContext());
    Status rc = global_context_->Init();
    if (rc.IsError()) {
      std::terminate();
    }
  });
  return global_context_.get();
}

Status GlobalContext::Init() {
  config_manager_ = std::make_shared<ConfigManager>();
  mem_pool_ = std::make_shared<SystemPool>();
  // For testing we can use Dummy pool instead

  // Create some tensor allocators for the different types and hook them into the pool.
  tensor_allocator_ = std::make_unique<Allocator<Tensor>>(mem_pool_);
  cv_tensor_allocator_ = std::make_unique<Allocator<CVTensor>>(mem_pool_);
  int_allocator_ = std::make_unique<IntAlloc>(mem_pool_);
  return Status::OK();
}

// A print method typically used for debugging
void GlobalContext::Print(std::ostream &out) const {
  out << "GlobalContext contains the following default config: " << *config_manager_ << "\n";
}
}  // namespace dataset
}  // namespace mindspore
