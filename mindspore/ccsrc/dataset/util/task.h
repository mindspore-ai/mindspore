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
#ifndef DATASET_UTIL_TASK_H_
#define DATASET_UTIL_TASK_H_

#include <chrono>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include "dataset/util/de_error.h"
#include "dataset/util/intrp_resource.h"
#include "dataset/util/list.h"
#include "dataset/util/memory_pool.h"
#include "dataset/util/services.h"
#include "dataset/util/wait_post.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
class TaskManager;

class Task : public IntrpResource {
 public:
  friend class TaskManager;

  friend class TaskGroup;

  Task(const std::string &myName, const std::function<Status()> &f);

  // Future objects are not copyable.
  Task(const Task &) = delete;

  ~Task() override;

  Task &operator=(const Task &) = delete;

  // Move constructor and Assignment are not supported.
  // Too many things in this class.
  Task(Task &&) = delete;

  Task &operator=(Task &&) = delete;

  Status GetTaskErrorIfAny();

  void ChangeName(const std::string &newName) { my_name_ = newName; }

  // To execute the _fncObj
  void operator()();

  Node<Task> node;
  Node<Task> group;
  Node<Task> free;

  // Run the task
  Status Run();

  Status Join();

  bool Running() const { return running_; }

  bool CaughtSevereException() const { return caught_severe_exception_; }

  bool IsMasterThread() const { return is_master_; }

  std::thread::id get_id() { return id_; }

  std::string MyName() { return my_name_; }

  // An operator used by std::find
  bool operator==(const Task &other) const { return (this == &other); }

  bool operator!=(const Task &other) const { return !(*this == other); }

  void Post() { wp_.Set(); }

  Status Wait() { return (wp_.Wait()); }

  void set_task_group(TaskGroup *vg);

 private:
  std::mutex mux_;
  std::string my_name_;
  Status rc_;
  WaitPost wp_;
  // Task need to provide definition for this function.  It
  // will be called by thread function.
  std::function<Status()> fnc_obj_;
  // Misc fields used by TaskManager.
  TaskGroup *task_group_;
  std::future<void> thrd_;
  std::thread::id id_;
  bool is_master_;
  volatile bool running_;
  volatile bool caught_severe_exception_;

  void ShutdownGroup();
  TaskGroup *MyTaskGroup();
};

extern thread_local Task *gMyTask;
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_TASK_H_
