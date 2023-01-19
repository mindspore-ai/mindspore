/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <functional>
#include <set>
#include "./securec.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
TaskManager *TaskManager::instance_ = nullptr;
std::once_flag TaskManager::init_instance_flag_;
// This takes the same parameter as Task constructor.
Status TaskManager::CreateAsyncTask(const std::string &my_name, const std::function<Status()> &f, TaskGroup *vg,
                                    Task **task, int32_t operator_id) {
  // We need to block destructor coming otherwise we will deadlock. We will grab the
  // stateLock in shared allowing CreateAsyncTask to run concurrently.
  SharedLock stateLck(&state_lock_);
  // Now double check the state
  if (ServiceState() == STATE::kStopInProg || ServiceState() == STATE::kStopped) {
    RETURN_STATUS_ERROR(StatusCode::kMDInterrupted, "TaskManager is shutting down");
  }
  RETURN_IF_NOT_OK(GetFreeTask(my_name, f, task, operator_id));
  if (vg == nullptr) {
    RETURN_STATUS_UNEXPECTED("TaskGroup is null");
  }
  // Previously there is a timing hole where the thread is spawn but hit error immediately before we can set
  // the TaskGroup pointer. We will do the set here before we call run(). The run() will do the registration.
  (*task)->set_task_group(vg);
  // Link to the master lru list.
  {
    UniqueLock lck(&lru_lock_);
    lru_.Append(*task);
  }
  // Link to the group list as well before we spawn.
  {
    UniqueLock lck(&vg->rw_lock_);
    vg->grp_list_.Append(*task);
  }
  // Track all the TaskGroup. Used for control-c
  {
    LockGuard lck(&tg_lock_);
    (void)this->grp_list_.insert(vg);
  }
  RETURN_IF_NOT_OK((*task)->wp_.Register(vg));
  RETURN_IF_NOT_OK((*task)->Run());
  // Wait for the thread to initialize successfully.
  RETURN_IF_NOT_OK((*task)->Wait());
  (*task)->Clear();
  return Status::OK();
}

Status TaskManager::join_all() {
  Status rc;
  Status rc2;
  SharedLock lck(&lru_lock_);
  for (Task &tk : lru_) {
    rc = tk.Join();
    if (rc.IsError()) {
      rc2 = rc;
    }
  }
  return rc2;
}

void TaskManager::interrupt_all() noexcept {
  global_interrupt_ = 1;
  LockGuard lck(&tg_lock_);
  for (TaskGroup *vg : grp_list_) {
    auto svc = vg->GetIntrpService();
    if (svc) {
      // Stop the interrupt service. No new request is accepted.
      Status rc = svc->ServiceStop();
      if (rc.IsError()) {
        MS_LOG(ERROR) << "Error while stopping the service. Message: " << rc;
      }
      svc->InterruptAll();
    }
  }
  master_->Interrupt();
}

Task *TaskManager::FindMe() {
#if !defined(_WIN32) && !defined(_WIN64)
  return gMyTask;
#else
  TaskManager &tm = TaskManager::GetInstance();
  SharedLock lock(&tm.lru_lock_);
  auto id = this_thread::get_id();
  for (auto iter = tm.lru_.begin(); iter != tm.lru_.end(); ++iter) {
    if (iter->id_ == id && iter->running_) {
      return &(*iter);
    }
  }
  // If we get here, either I am the watchdog or the master thread.
  if (tm.master_->id_ == id) {
    return tm.master_.get();
  } else if (tm.watchdog_ != nullptr && tm.watchdog_->id_ == id) {
    return tm.watchdog_;
  }
  MS_LOG(ERROR) << "Task not found.";
  return nullptr;
#endif
}

TaskManager::TaskManager() try : global_interrupt_(0),
                                 lru_(&Task::node),
                                 free_lst_(&Task::free),
                                 watchdog_grp_(nullptr),
                                 watchdog_(nullptr) {
  auto alloc = Services::GetAllocator<Task>();
  // Create a dummy Task for the master thread (this thread)
  master_ = std::allocate_shared<Task>(alloc, "master", []() -> Status { return Status::OK(); });
  master_->id_ = this_thread::get_id();
  master_->running_ = true;
  master_->is_master_ = true;
#if !defined(_WIN32) && !defined(_WIN64)
  gMyTask = master_.get();
#if !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  // Initialize the semaphore for the watchdog
  errno_t rc = sem_init(&sem_, 0, 0);
  if (rc == -1) {
    MS_LOG(ERROR) << "Unable to initialize a semaphore. Errno = " << rc << ".";
    std::terminate();
  }
#endif
#endif
} catch (const std::exception &e) {
  MS_LOG(ERROR) << "MindData initialization failed: " << e.what() << ".";
  std::terminate();
}

TaskManager::~TaskManager() {
  if (watchdog_) {
    WakeUpWatchDog();
    (void)watchdog_->Join();
    // watchdog_grp_ and watchdog_ pointers come from Services::GetInstance().GetServiceMemPool() which we will free it
    // on shutdown. So no need to free these pointers one by one.
    watchdog_grp_ = nullptr;
    watchdog_ = nullptr;
  }
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  (void)sem_destroy(&sem_);
#endif
}

Status TaskManager::DoServiceStart() {
  MS_LOG(INFO) << "Starting Task Manager.";
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  // Create a watchdog for control-c
  std::shared_ptr<MemoryPool> mp = Services::GetInstance().GetServiceMemPool();
  // A dummy group just for the watchdog. We aren't really using it. But most code assumes a thread must
  // belong to a group.
  auto f = std::bind(&TaskManager::WatchDog, this);
  Status rc;
  watchdog_grp_ = new (&rc, mp) TaskGroup();
  RETURN_IF_NOT_OK(rc);
  rc = watchdog_grp_->CreateAsyncTask("Watchdog", f, &watchdog_);
  if (rc.IsError()) {
    ::operator delete(watchdog_grp_, mp);
    watchdog_grp_ = nullptr;
    return rc;
  }
  (void)grp_list_.erase(watchdog_grp_);
  lru_.Remove(watchdog_);
#endif
  return Status::OK();
}

Status TaskManager::DoServiceStop() {
  WakeUpWatchDog();
  interrupt_all();
  return Status::OK();
}

Status TaskManager::WatchDog() {
  TaskManager::FindMe()->Post();
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  errno_t err = sem_wait(&sem_);
  if (err == -1) {
    RETURN_STATUS_UNEXPECTED("Errno = " + std::to_string(errno));
  }
  // We are woken up by control-c and we are going to stop all threads that are running.
  // In addition, we also want to prevent new thread from creating. This can be done
  // easily by calling the parent function.
  RETURN_IF_NOT_OK(ServiceStop());
#endif
  return Status::OK();
}

// Follow the group link and interrupt other
// Task in the same group. It is used by
// Watchdog only.
void TaskManager::InterruptGroup(Task &curTk) {
  TaskGroup *vg = curTk.MyTaskGroup();
  vg->interrupt_all();
}

void TaskManager::InterruptMaster(const Status &rc) {
  TaskManager &tm = TaskManager::GetInstance();
  std::shared_ptr<Task> master = tm.master_;
  std::lock_guard<std::mutex> lck(master->mux_);
  master->Interrupt();
  if (rc.IsError() && master->rc_.IsOk()) {
    master->rc_ = rc;
    master->caught_severe_exception_ = true;
    // Move log error here for some scenarios didn't call GetMasterThreadRc
    if (master->rc_.StatusCode() != mindspore::StatusCode::kMDPyFuncException) {
      // use python operation, the error had been raised in python layer. So disable log prompt here.
      MS_LOG(ERROR) << "Task is terminated with err msg (more details are in info level logs): " << master->rc_;
    }
  }
}

Status TaskManager::GetMasterThreadRc() {
  TaskManager &tm = TaskManager::GetInstance();
  std::shared_ptr<Task> master = tm.master_;
  Status rc = tm.master_->GetTaskErrorIfAny();
  if (rc.IsError()) {
    // Reset the state once we retrieve the value.
    std::lock_guard<std::mutex> lck(master->mux_);
    master->rc_ = Status::OK();
    master->caught_severe_exception_ = false;
    master->ResetIntrpState();
  }
  return rc;
}

void TaskManager::ReturnFreeTask(Task *p) noexcept {
  // Take it out from lru_ if any
  {
    UniqueLock lck(&lru_lock_);
    auto iter = lru_.begin();
    for (; iter != lru_.end(); ++iter) {
      if (*iter == *p) {
        break;
      }
    }
    if (iter != lru_.end()) {
      lru_.Remove(p);
    }
  }
  // We need to deallocate the string resources associated with the Task class
  // before we cache its memory for future use.
  p->~Task();
  // Put it back into free list
  {
    LockGuard lck(&free_lock_);
    free_lst_.Append(p);
  }
}

Status TaskManager::GetFreeTask(const std::string &my_name, const std::function<Status()> &f, Task **p,
                                int32_t operator_id) {
  if (p == nullptr) {
    RETURN_STATUS_UNEXPECTED("p is null");
  }
  Task *q = nullptr;
  // First try the free list
  {
    LockGuard lck(&free_lock_);
    if (free_lst_.count > 0) {
      q = free_lst_.head;
      free_lst_.Remove(q);
    }
  }
  if (q) {
    new (q) Task(my_name, f, operator_id);
  } else {
    std::shared_ptr<MemoryPool> mp = Services::GetInstance().GetServiceMemPool();
    Status rc;
    q = new (&rc, mp) Task(my_name, f, operator_id);
    RETURN_IF_NOT_OK(rc);
  }
  *p = q;
  return Status::OK();
}

Status TaskGroup::CreateAsyncTask(const std::string &my_name, const std::function<Status()> &f, Task **ppTask,
                                  int32_t operator_id) {
  auto pMytask = TaskManager::FindMe();
  // We need to block ~TaskGroup coming otherwise we will deadlock. We will grab the
  // stateLock in shared allowing CreateAsyncTask to run concurrently.
  SharedLock state_lck(&state_lock_);
  // Now double check the state
  if (ServiceState() != STATE::kRunning) {
    RETURN_STATUS_ERROR(StatusCode::kMDInterrupted, "Taskgroup is shutting down");
  }
  TaskManager &dm = TaskManager::GetInstance();
  Task *pTask = nullptr;
  // If the group is already in error, early exit too.
  // We can't hold the rc_mux_ throughout because the thread spawned by CreateAsyncTask may hit error which
  // will try to shutdown the group and grab the rc_mux_ and we will deadlock.
  {
    std::unique_lock<std::mutex> rcLock(rc_mux_);
    if (rc_.IsError()) {
      return pMytask->IsMasterThread() ? rc_ : Status(StatusCode::kMDInterrupted);
    }
  }
  RETURN_IF_NOT_OK(dm.CreateAsyncTask(my_name, f, this, &pTask, operator_id));
  if (ppTask) {
    *ppTask = pTask;
  }
  return Status::OK();
}

void TaskGroup::interrupt_all() noexcept {
  // There is a racing condition if we don't stop the interrupt service at this point. New resource
  // may come in and not being picked up after we call InterruptAll(). So stop new comers and then
  // interrupt any existing resources.
  (void)intrp_svc_->ServiceStop();
  intrp_svc_->InterruptAll();
}

Status TaskGroup::join_all(Task::WaitFlag wf) {
  Status rc;
  Status rc2;
  SharedLock lck(&rw_lock_);
  for (Task &tk : grp_list_) {
    rc = tk.Join(wf);
    if (rc.IsError()) {
      rc2 = rc;
    }
  }
  return rc2;
}

Status TaskGroup::DoServiceStop() {
  interrupt_all();
  return (join_all(Task::WaitFlag::kNonBlocking));
}

TaskGroup::TaskGroup() : grp_list_(&Task::group), intrp_svc_(nullptr) {
  auto alloc = Services::GetAllocator<IntrpService>();
  intrp_svc_ = std::allocate_shared<IntrpService>(alloc);
  (void)Service::ServiceStart();
}

TaskGroup::~TaskGroup() {
  (void)Service::ServiceStop();
  // The TaskGroup is going out of scope, and we can return the Task list to the free list.
  Task *cur = grp_list_.head;
  TaskManager &tm = TaskManager::GetInstance();
  while (cur) {
    Task *next = cur->group.next;
    grp_list_.Remove(cur);
    tm.ReturnFreeTask(cur);
    cur = next;
  }
  {
    LockGuard lck(&tm.tg_lock_);
    (void)tm.grp_list_.erase(this);
  }
}

Status TaskGroup::GetTaskErrorIfAny() {
  SharedLock lck(&rw_lock_);
  for (Task &tk : grp_list_) {
    RETURN_IF_NOT_OK(tk.GetTaskErrorIfAny());
  }
  return Status::OK();
}

std::shared_ptr<IntrpService> TaskGroup::GetIntrpService() { return intrp_svc_; }
}  // namespace dataset
}  // namespace mindspore
