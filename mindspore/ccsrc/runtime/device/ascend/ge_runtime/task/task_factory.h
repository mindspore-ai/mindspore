/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_TASK_FACTORY_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_TASK_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include "runtime/device/ascend/ge_runtime/task_info.h"
#include "mindspore/core/utils/log_adapter.h"

namespace mindspore::ge::model_runner {
class Task;
class ModelContext;
using TASK_CREATOR_FUN = std::function<std::shared_ptr<Task>(const ModelContext &, std::shared_ptr<TaskInfo>)>;

class TaskFactory {
 private:
  TaskFactory() {}
  ~TaskFactory() {}
  void RegisterCreator(const TaskInfoType &type, const TASK_CREATOR_FUN &func) {
    if (creator_map_.find(type) != creator_map_.end()) {
      MS_LOG(WARNING) << "Creator type " << type << " already exist.";
    }
    creator_map_[type] = func;
  }

  std::map<TaskInfoType, TASK_CREATOR_FUN> creator_map_;

 public:
  static TaskFactory &GetInstance() {
    static TaskFactory instance;
    return instance;
  }

  std::shared_ptr<Task> Create(const ModelContext &model_context, const std::shared_ptr<TaskInfo> &task_info) const {
    if (task_info == nullptr) {
      MS_LOG(ERROR) << "task_info is null.";
      return nullptr;
    }

    auto iter = creator_map_.find(task_info->type());
    if (iter == creator_map_.end()) {
      MS_LOG(ERROR) << "Unknown task type " << task_info->type();
      return nullptr;
    }
    return iter->second(model_context, task_info);
  }

  class Register {
   public:
    Register(const TaskInfoType &type, const TASK_CREATOR_FUN &func) {
      MS_LOG(DEBUG) << "register type " << type;
      TaskFactory::GetInstance().RegisterCreator(type, func);
    }

    ~Register() {}
  };
};

#define REGISTER_TASK(type, task_clazz, task_info_clazz)                                                               \
  TaskFactory::Register g_##task_clazz##_register(                                                                     \
    type, [](const ModelContext &model_context, const std::shared_ptr<TaskInfo> &task_info) -> std::shared_ptr<Task> { \
      std::shared_ptr<task_info_clazz> concrete_task_info = std::static_pointer_cast<task_info_clazz>(task_info);      \
      return std::make_shared<task_clazz>(model_context, concrete_task_info);                                          \
    });

}  // namespace mindspore::ge::model_runner
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_TASK_FACTORY_H_
