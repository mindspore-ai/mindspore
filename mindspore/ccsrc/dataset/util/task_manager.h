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
#ifndef DATASET_UTIL_TASK_MANAGER_H_
#define DATASET_UTIL_TASK_MANAGER_H_

#if !defined(_WIN32) && !defined(_WIN64)
#include <semaphore.h>
#include <signal.h>  // for sig_atomic_t
#endif
#include <condition_variable>
#include <functional>
#include <memory>
#include <string>
#include <set>
#include "dataset/util/allocator.h"
#include "dataset/util/intrp_service.h"
#include "dataset/util/lock.h"
#include "dataset/util/services.h"
#include "dataset/util/status.h"
#include "dataset/util/task.h"

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

  static TaskManager &GetInstance() noexcept { return Services::getTaskMgrInstance(); }

  Status DoServiceStart() override;

  Status DoServiceStop() override;

  // A public global interrupt flag for signal handlers
  volatile sig_atomic_t global_interrupt_;

  // API
  // This takes the same parameter as Task constructor. Take a look
  // of the test-thread.cc for usage.
  Status CreateAsyncTask(const std::string &my_name, const std::function<Status()> &f, TaskGroup *vg, Task **);

  // Same usage as boot thread group
  Status join_all();

  void interrupt_all() noexcept;

  // Locate a particular Task.
  static Task *FindMe();

  static void InterruptGroup(Task &);

  static Status GetMasterThreadRc();

  static void InterruptMaster(const Status &rc = Status::OK());

  static void WakeUpWatchDog() {
#if !defined(_WIN32) && !defined(_WIN64)
    TaskManager &tm = TaskManager::GetInstance();
    (void)sem_post(&tm.sem_);
#endif
  }

  void ReturnFreeTask(Task *p) noexcept;

  Status GetFreeTask(const std::string &my_name, const std::function<Status()> &f, Task **p);

  Status WatchDog();

 private:
  RWLock lru_lock_;
  SpinLock free_lock_;
  SpinLock tg_lock_;
  std::shared_ptr<Task> master_;
  List<Task> lru_;
  List<Task> free_lst_;
#if !defined(_WIN32) && !defined(_WIN64)
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

  Status CreateAsyncTask(const std::string &my_name, const std::function<Status()> &f, Task **pTask = nullptr);

  void interrupt_all() noexcept;

  Status join_all();

  int size() const noexcept { return grp_list_.count; }

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
  return (my_task != nullptr) ? my_task->Interrupted() : false;
}
}  // namespace this_thread

#define RETURN_IF_INTERRUPTED()                                          \
  do {                                                                   \
    if (mindspore::dataset::this_thread::is_interrupted()) {             \
      Task *myTask = TaskManager::FindMe();                              \
      if (myTask->IsMasterThread() && myTask->CaughtSevereException()) { \
        return TaskManager::GetMasterThreadRc();                         \
      } else {                                                           \
        return Status(StatusCode::kInterrupted);                         \
      }                                                                  \
    }                                                                    \
  } while (false)

inline Status interruptible_wait(std::condition_variable *cv, std::unique_lock<std::mutex> *lk,
                                 const std::function<bool()> &pred) noexcept {
  if (!pred()) {
    do {
      RETURN_IF_INTERRUPTED();
      try {
        (void)cv->wait_for(*lk, std::chrono::milliseconds(1));
      } catch (std::exception &e) {
        // Anything thrown by wait_for is considered system error.
        RETURN_STATUS_UNEXPECTED(e.what());
      }
    } while (!pred());
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_TASK_MANAGER_H_
