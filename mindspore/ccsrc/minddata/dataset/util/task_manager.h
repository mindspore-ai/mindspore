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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_TASK_MANAGER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_TASK_MANAGER_H_

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <semaphore.h>
#include <signal.h>  // for sig_atomic_t
#endif
#include <condition_variable>
#include <functional>
#include <memory>
#include <string>
#include <set>
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/intrp_service.h"
#include "minddata/dataset/util/lock.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/task.h"

namespace mindspore {
namespace dataset {
namespace thread {
using id = std::thread::id;
}  // namespace thread

namespace this_thread {
inline thread::id get_id() { return std::this_thread::get_id(); }
}  // namespace this_thread

class TaskManager : public Service {
 public:
  friend class Services;

  friend class TaskGroup;

  ~TaskManager() override;

  TaskManager(const TaskManager &) = delete;

  TaskManager &operator=(const TaskManager &) = delete;

  static Status CreateInstance() {
    std::call_once(init_instance_flag_, [&]() -> Status {
      auto &svcManager = Services::GetInstance();
      RETURN_IF_NOT_OK(svcManager.AddHook(&instance_));
      return Status::OK();
    });
    return Status::OK();
  }

  static TaskManager &GetInstance() noexcept { return *instance_; }

  Status DoServiceStart() override;

  Status DoServiceStop() override;

  // A public global interrupt flag for signal handlers
  volatile sig_atomic_t global_interrupt_;

  // API
  // This takes the same parameter as Task constructor. Take a look
  // of the test-thread.cc for usage.
  Status CreateAsyncTask(const std::string &my_name, const std::function<Status()> &f, TaskGroup *vg, Task **,
                         int32_t operator_id = -1);

  // Same usage as boot thread group
  Status join_all();

  void interrupt_all() noexcept;

  // Locate a particular Task.
  static Task *FindMe();

  static void InterruptGroup(Task &);

  static Status GetMasterThreadRc();

  static void InterruptMaster(const Status &rc = Status::OK());

  static void WakeUpWatchDog() {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
    TaskManager &tm = TaskManager::GetInstance();
    (void)sem_post(&tm.sem_);
#endif
  }

  void ReturnFreeTask(Task *p) noexcept;

  Status GetFreeTask(const std::string &my_name, const std::function<Status()> &f, Task **p, int32_t operator_id = -1);

  Status WatchDog();

 private:
  static std::once_flag init_instance_flag_;
  static TaskManager *instance_;
  RWLock lru_lock_;
  SpinLock free_lock_;
  SpinLock tg_lock_;
  std::shared_ptr<Task> master_;
  List<Task> lru_;
  List<Task> free_lst_;
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  sem_t sem_;
#endif
  TaskGroup *watchdog_grp_;
  std::set<TaskGroup *> grp_list_;
  Task *watchdog_;

  TaskManager();
};

// A group of related tasks.
class TaskGroup : public Service {
 public:
  friend class Task;
  friend class TaskManager;

  Status CreateAsyncTask(const std::string &my_name, const std::function<Status()> &f, Task **pTask = nullptr,
                         int32_t operator_id = -1);

  void interrupt_all() noexcept;

  Status join_all(Task::WaitFlag wf = Task::WaitFlag::kBlocking);

  int size() const noexcept { return grp_list_.count; }

  List<Task> GetTask() const noexcept { return grp_list_; }

  Status DoServiceStart() override { return Status::OK(); }

  Status DoServiceStop() override;

  TaskGroup();

  ~TaskGroup() override;

  Status GetTaskErrorIfAny();

  std::shared_ptr<IntrpService> GetIntrpService();

 private:
  Status rc_;
  // Can't use rw_lock_ as we will lead to deadlatch. Create another mutex to serialize access to rc_.
  std::mutex rc_mux_;
  RWLock rw_lock_;
  List<Task> grp_list_;
  std::shared_ptr<IntrpService> intrp_svc_;
};

namespace this_thread {
inline bool is_interrupted() {
  TaskManager &tm = TaskManager::GetInstance();
  if (tm.global_interrupt_ == 1) {
    return true;
  }
  Task *my_task = TaskManager::FindMe();
  return my_task->Interrupted();
}

inline bool is_master_thread() {
  Task *my_task = TaskManager::FindMe();
  return my_task->IsMasterThread();
}

inline Status GetInterruptStatus() {
  Task *my_task = TaskManager::FindMe();
  return my_task->GetInterruptStatus();
}
}  // namespace this_thread

#define RETURN_IF_INTERRUPTED()                                            \
  do {                                                                     \
    if (mindspore::dataset::this_thread::is_interrupted()) {               \
      return Task::OverrideInterruptRc(this_thread::GetInterruptStatus()); \
    }                                                                      \
  } while (false)

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_TASK_MANAGER_H_
