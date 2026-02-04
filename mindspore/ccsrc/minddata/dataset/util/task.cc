/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/util/thread_bind_core_utils.h"
#ifdef WITH_BACKEND
#include "utils/ms_context.h"
#include "include/runtime/hardware_abstract/data_queue/data_queue_mgr.h"
#endif
namespace mindspore {
namespace dataset {
thread_local Task *gMyTask = nullptr;

void Task::operator()() {
  mindspore::dataset::BindThreadCoreForMindDataOp(my_name_);
#if !defined(_WIN32) && !defined(_WIN64)
  gMyTask = this;
#endif
  id_ = this_thread::get_id();
  std::stringstream ss;
  ss << id_;
#if defined(__APPLE__)
  // The thread id in Linux may be duplicate
  ss << Services::GetUniqueID();
#endif
  MS_LOG(DEBUG) << "Task: " << my_name_ << " Thread ID " << ss.str() << " Started.";

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  native_handle_ = pthread_self();
  thread_id_ = syscall(SYS_gettid);
#endif

  try {
    // Previously there is a timing hole where the thread is spawn but hit error immediately before we can set
    // the TaskGroup pointer and register. We move the registration logic to here (after we spawn) so we can
    // get the thread id.
    TaskGroup *vg = MyTaskGroup();
    if (vg == nullptr) {
      MS_LOG(ERROR) << "Task Group is nullptr.";
      ShutdownGroup();
      return;
    }
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
  } catch (const std::bad_alloc &e) {
    rc_ = STATUS_ERROR(StatusCode::kMDOutOfMemory, e.what());
    MS_LOG(ERROR) << rc_;
    ShutdownGroup();
  } catch (const std::exception &e) {
    rc_ = STATUS_ERROR(StatusCode::kMDUnexpectedError, e.what());
    MS_LOG(INFO) << rc_;
    ShutdownGroup();
  }
  // The given function has finished running. We must change the running status immediately.
  // Because std::async may create a new thread with the same thread ID as this thread since it has finished.
  // Then there will be two tasks with the same thread ID in our task group, which may cause a mismatch
  // in TaskManager::FindMe(). We can identify the exact task based on the running status there.
  running_ = false;
  MS_LOG(DEBUG) << "Task: " << my_name_ << " Thread ID " << ss.str() << " Finished.";
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
        if (vg->rc_.IsError()) {
          // InterruptMaster miss sink pyfunc scenario, thus add print here.
          if (vg->has_dataqueue_ && vg->rc_.StatusCode() == mindspore::StatusCode::kMDPyFuncException) {
            MS_LOG(ERROR) << "MindSpore dataset is terminated with err msg: " << vg->rc_;
          }
        }
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

pid_t GetCurrentPID() {
#if defined(_WIN32) || defined(_WIN64)
  return GetCurrentProcessId();
#else
  return getpid();
#endif
}

Task::Task(const std::string &myName, const std::function<Status()> &f, int32_t operator_id)
    : my_name_(myName),
      operator_id_(operator_id),
      process_id_(GetCurrentPID()),
      thread_id_(-1),
      rc_(Status::OK()),
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
      running_ = true;
      thrd_ = std::async(std::launch::async, std::ref(*this));
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
  // If the current process is a subprocess of map or batch, the process ID will not be equal to process_id_.
  // And no need to join WatchDog.
  if (running_ && GetCurrentPID() == process_id_ && my_name_.find("WatchDog") == std::string::npos) {
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
        const uint32_t kLogInterval = 5;
        while (thrd_.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
          // We can't tell which conditional_variable this thread is waiting on. So we may need
          // to interrupt everything one more time.
          std::stringstream ss;
          ss << get_id();
          wait_times++;
          if (wait_times % kLogInterval == 0) {
            MS_LOG(WARNING) << "Task: " << my_name_ << " Thread ID " << ss.str()
                            << " is not finished and cannot be joined. Try to interrupt again.";
          }
          interrupt_svc->InterruptAll();
#ifdef WITH_BACKEND
          const int kMaxWaitTimes = 5;
          if (device_target == kAscendDevice) {
            // Because hostPush hung in DataQueueOp, wait 5 seconds and destroy the tdt
            if (wait_times > kMaxWaitTimes && my_name_.find("DataQueueOp") != std::string::npos) {
              MS_LOG(WARNING) << "Wait " << wait_times << " seconds, the task: " << my_name_
                              << " will be destroyed by TdtHostDestory.";
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

          // Because ReceiveBridgeOp maybe hung by MsgRcv from SendBridgeOp
          if (wait_times > kMaxWaitTimes && my_name_.find("ReceiveBridgeOp") != std::string::npos) {
            MS_LOG(WARNING) << "Wait " << wait_times << " seconds, the task: " << my_name_ << ".";

            // just wait 30 seconds
            if (wait_times > kWaitInterruptTaskTime) {
              MS_LOG(WARNING) << "Task: " << my_name_ << " Thread ID " << ss.str()
                              << " is not responding. Break the interrupt.";
              break;
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

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
pthread_t Task::GetNativeHandle() const { return native_handle_; }
#endif

}  // namespace dataset
}  // namespace mindspore
