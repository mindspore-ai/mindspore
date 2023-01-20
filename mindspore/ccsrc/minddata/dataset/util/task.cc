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
#include "minddata/dataset/util/task.h"
#include "utils/os.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/task_manager.h"
#if defined(__ANDROID__) || defined(ANDROID)
#include "minddata/dataset/util/services.h"
#endif
#ifdef WITH_BACKEND
#include "utils/ms_context.h"
#include "mindspore/ccsrc/include/backend/data_queue/data_queue_mgr.h"
#endif
namespace mindspore {
namespace dataset {
thread_local Task *gMyTask = nullptr;

void Task::operator()() {
#if !defined(_WIN32) && !defined(_WIN64)
  gMyTask = this;
#endif
  id_ = this_thread::get_id();
  std::stringstream ss;
  ss << id_;
#if defined(__ANDROID__) || defined(ANDROID) || defined(__APPLE__)
  // The thread id in Linux may be duplicate
  ss << Services::GetUniqueID();
#endif
  MS_LOG(DEBUG) << "Task: " << my_name_ << " Thread ID " << ss.str() << " Started.";

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  native_handle_ = pthread_self();
  thread_id_ = syscall(SYS_gettid);
#endif
  try {
    // Previously there is a timing hole where the thread is spawn but hit error immediately before we can set
    // the TaskGroup pointer and register. We move the registration logic to here (after we spawn) so we can
    // get the thread id.
    TaskGroup *vg = MyTaskGroup();
    std::string uuid = ss.str();
    auto intrp_service = vg->GetIntrpService();
    rc_ = intrp_service->Register(&uuid, this);
    if (rc_.IsOk()) {
      // Now we can run the given task.
      rc_ = fnc_obj_();
    }
    // Some error codes are ignored, e.g. interrupt. Others we just shutdown the group.
    if (rc_.IsError() && rc_ != StatusCode::kMDInterrupted) {
      if (rc_.StatusCode() == StatusCode::kMDNetWorkError) {
        MS_LOG(WARNING) << rc_;
      } else {
        MS_LOG(INFO) << "Task: " << my_name_ << " - thread(" << uuid << ") is terminated with err msg: " << rc_;
      }
      ShutdownGroup();
    }
    // The given function has finished running. We must change the running status immediately.
    // Because std::async may create a new thread with the same thread ID as this thread since it has finished.
    // Then there will be two tasks with the same thread ID in our task group, which may cause a mismatch
    // in TaskManager::FindMe(). We can identify the exact task based on the running status there.
    running_ = false;
    MS_LOG(DEBUG) << "Task: " << my_name_ << " Thread ID " << ss.str() << " Finished.";
  } catch (const std::bad_alloc &e) {
    rc_ = STATUS_ERROR(StatusCode::kMDOutOfMemory, e.what());
    MS_LOG(ERROR) << rc_;
    ShutdownGroup();
  } catch (const std::exception &e) {
    rc_ = STATUS_ERROR(StatusCode::kMDUnexpectedError, e.what());
    MS_LOG(ERROR) << rc_;
    ShutdownGroup();
  }
}

void Task::ShutdownGroup() {  // Wake up watch dog and shutdown the engine.
  {
    std::lock_guard<std::mutex> lk(mux_);
    caught_severe_exception_ = true;
  }
  TaskGroup *vg = MyTaskGroup();
  // If multiple threads hit severe errors in the same group. Keep the first one and
  // discard the rest.
  std::unique_lock<std::mutex> rcLock(vg->rc_mux_);
  {
    if (vg->rc_.IsOk()) {
      // Check again after we get the lock
      if (vg->rc_.IsOk()) {
        vg->rc_ = rc_;
        rcLock.unlock();
        TaskManager::InterruptMaster(rc_);
        TaskManager::InterruptGroup(*this);
      }
    }
  }
}

Status Task::GetTaskErrorIfAny() const {
  std::lock_guard<std::mutex> lk(mux_);
  if (caught_severe_exception_) {
    return rc_;
  } else {
    return Status::OK();
  }
}

Task::Task(const std::string &myName, const std::function<Status()> &f, int32_t operator_id)
    : my_name_(myName),
      operator_id_(operator_id),
      thread_id_(-1),
      rc_(),
      fnc_obj_(f),
      task_group_(nullptr),
      is_master_(false),
      running_(false),
      caught_severe_exception_(false),
      native_handle_(0) {
  IntrpResource::ResetIntrpState();
  wp_.ResetIntrpState();
  wp_.Clear();
}

Status Task::Run() {
  Status rc;
  std::lock_guard<std::mutex> lk(mux_);
  if (running_ == false) {
    try {
      thrd_ = std::async(std::launch::async, std::ref(*this));
      running_ = true;
      caught_severe_exception_ = false;
    } catch (const std::exception &e) {
      rc = STATUS_ERROR(StatusCode::kMDUnexpectedError, e.what());
    }
  }
  return rc;
}

Status Task::Join(WaitFlag blocking) {
#ifdef WITH_BACKEND
  RETURN_UNEXPECTED_IF_NULL(MsContext::GetInstance());
  std::string device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
#endif
  if (running_) {
    RETURN_UNEXPECTED_IF_NULL(MyTaskGroup());
    auto interrupt_svc = MyTaskGroup()->GetIntrpService();
    try {
      if (blocking == WaitFlag::kBlocking) {
        // If we are asked to wait, then wait
        thrd_.get();
      } else if (blocking == WaitFlag::kNonBlocking) {
        // There is a race condition in the global resource tracking such that a thread can miss the
        // interrupt and becomes blocked on a conditional variable forever. As a result, calling
        // join() will not come back. We need some timeout version of join such that if the thread
        // doesn't come back in a reasonable of time, we will send the interrupt again.
        uint32_t wait_times = 0;
        while (thrd_.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
          // We can't tell which conditional_variable this thread is waiting on. So we may need
          // to interrupt everything one more time.
          std::stringstream ss;
          ss << get_id();
          MS_LOG(WARNING) << "Task: " << my_name_ << " Thread ID " << ss.str()
                          << " is not responding. Interrupt it again.";
          interrupt_svc->InterruptAll();
          wait_times++;
#ifdef WITH_BACKEND
          if (device_target == kAscendDevice) {
            // Because hostPush hung in DataQueueOp, wait 5 seconds and destroy the tdt
            if (wait_times > 5 && my_name_.find("DataQueueOp") != std::string::npos) {
              MS_LOG(WARNING) << "Wait " << wait_times << " seconds, "
                              << "the task: " << my_name_ << " will be destroyed by TdtHostDestory.";
              if (device::DataQueueMgr::DestoryTdtHandle()) {
                MS_LOG(INFO) << "Destroy tdt channel success.";
              } else {
                MS_LOG(WARNING) << "Destroy tdt channel failed.";
              }

              // just wait 30 seconds
              // case1: cpu usage 100%, DataQueueOp thread may destroy without thread_future
              if (wait_times > kWaitInterruptTaskTime) {
                MS_LOG(WARNING) << "Task: " << my_name_ << " Thread ID " << ss.str()
                                << " is not responding. Maybe it has been destroyed. Stop the task.";
                break;
              }
            }
          }
#endif
        }
      } else {
        RETURN_STATUS_UNEXPECTED("Unknown WaitFlag");
      }
      std::stringstream ss;
      ss << get_id();
      MS_LOG(DEBUG) << "Task: " << my_name_ << " Thread ID " << ss.str() << " Stopped.";
      running_ = false;
      RETURN_IF_NOT_OK(wp_.Deregister());
      RETURN_IF_NOT_OK(interrupt_svc->Deregister(ss.str()));
    } catch (const std::exception &e) {
      RETURN_STATUS_UNEXPECTED(e.what());
    }
  }
  return Status::OK();
}

TaskGroup *Task::MyTaskGroup() { return task_group_; }

void Task::set_task_group(TaskGroup *vg) { task_group_ = vg; }

Task::~Task() { task_group_ = nullptr; }
Status Task::OverrideInterruptRc(const Status &rc) {
  if (rc == StatusCode::kMDInterrupted && this_thread::is_master_thread()) {
    // If we are interrupted, override the return value if this is the master thread.
    // Master thread is being interrupted mostly because of some thread is reporting error.
    return TaskManager::GetMasterThreadRc();
  }
  return rc;
}

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
pthread_t Task::GetNativeHandle() const { return native_handle_; }
#endif

}  // namespace dataset
}  // namespace mindspore
