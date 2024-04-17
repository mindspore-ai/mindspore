/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <cerrno>
#include <cstring>
#include <csignal>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include "mindrt/include/fork_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "mindrt/src/actor/actormgr.h"
#include "runtime/pynative/op_executor.h"
#include "pipeline/pynative/pynative_execute.h"
#include "pipeline/jit/ps/pipeline.h"
#include "include/common/thread_pool.h"
#include "include/common/pybind_api/api_register.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {

void RegisterForkCallbacks() {
#if !defined(_WIN32)
  MS_LOG(DEBUG) << "Register MsContext fork callbacks.";
  ForkUtils::GetInstance().RegisterCallbacks(MsContext::GetInstance(), static_cast<void (MsContext::*)()>(nullptr),
                                             static_cast<void (MsContext::*)()>(nullptr), &MsContext::ChildAfterFork);
  MS_LOG(DEBUG) << "Register ActorMgr fork callbacks.";
  ForkUtils::GetInstance().RegisterCallbacks(ActorMgr::GetActorMgrRef(), static_cast<void (ActorMgr::*)()>(nullptr),
                                             static_cast<void (ActorMgr::*)()>(nullptr), &ActorMgr::ChildAfterFork);
  MS_LOG(DEBUG) << "Register Common ThreadPool fork callbacks.";
  ForkUtils::GetInstance().RegisterCallbacks(
    &common::ThreadPool::GetInstance(), static_cast<void (common::ThreadPool::*)()>(nullptr),
    static_cast<void (common::ThreadPool::*)()>(nullptr), &common::ThreadPool::ChildAfterFork);
  MS_LOG(DEBUG) << "Register PyNativeExecutor fork callbacks.";
  ForkUtils::GetInstance().RegisterCallbacks(
    pynative::PyNativeExecutor::GetInstance(), &pynative::PyNativeExecutor::ParentBeforeFork,
    static_cast<void (pynative::PyNativeExecutor::*)()>(nullptr), &pynative::PyNativeExecutor::ChildAfterFork);
  MS_LOG(DEBUG) << "Register GraphExecutorPy fork callbacks.";
  ForkUtils::GetInstance().RegisterCallbacks(
    pipeline::GraphExecutorPy::GetInstance(), &pipeline::GraphExecutorPy::ParentBeforeFork,
    &pipeline::GraphExecutorPy::ParentAfterFork, &pipeline::GraphExecutorPy::ChildAfterFork);
  MS_LOG(DEBUG) << "Register OpExecutor fork callbacks.";
  ForkUtils::GetInstance().RegisterCallbacks(
    &runtime::OpExecutor::GetInstance(), static_cast<void (runtime::OpExecutor::*)()>(nullptr),
    static_cast<void (runtime::OpExecutor::*)()>(nullptr), &runtime::OpExecutor::ChildAfterFork);
  MS_LOG(DEBUG) << "Register DeviceContextManager fork callbacks.";
  ForkUtils::GetInstance().RegisterCallbacks(
    &device::DeviceContextManager::GetInstance(), static_cast<void (device::DeviceContextManager::*)()>(nullptr),
    static_cast<void (device::DeviceContextManager::*)()>(nullptr), &device::DeviceContextManager::ChildAfterFork);
  MS_LOG(DEBUG) << "Register GraphScheduler fork callbacks.";
  ForkUtils::GetInstance().RegisterCallbacks(
    &runtime::GraphScheduler::GetInstance(), static_cast<void (runtime::GraphScheduler::*)()>(nullptr),
    static_cast<void (runtime::GraphScheduler::*)()>(nullptr), &runtime::GraphScheduler::ChildAfterFork);
#endif
}

void PrepareBeforeFork() {
  MS_LOG(DEBUG) << "Parent process before fork.";

  // Register fork callbacks when first fork event occurs.
  static std::once_flag once_flag_;
  std::call_once(once_flag_, [&]() {
    MS_LOG(DEBUG) << "Register fork event callbacks.";
    RegisterForkCallbacks();
  });

  // Trigger ParentBeforeFork callbacks in parent process.
  ForkUtils::GetInstance().BeforeFork();

  // If the forked thread does not hold the gil lock, we need to manually acquire the gil lock before forking,
  // otherwise the child process will block when acquiring the gil lock.
  ForkUtils::GetInstance().SetGilHoldBeforeFork(PyGILState_Check());
  if (!ForkUtils::GetInstance().IsGilHoldBeforeFork()) {
    MS_LOG(DEBUG) << "Acquire GIL lock in parent process before fork.";
    ForkUtils::GetInstance().SetGilState(static_cast<int>(PyGILState_Ensure()));
  }
}

void ParentAtFork() {
  MS_LOG(DEBUG) << "Parent process at fork.";

  // Release the gil lock that was acquired manually before forking.
  if (!ForkUtils::GetInstance().IsGilHoldBeforeFork()) {
    MS_LOG(DEBUG) << "Release GIL lock acquired manually before fork.";
    PyGILState_Release(static_cast<PyGILState_STATE>(ForkUtils::GetInstance().GetGilState()));
  }

  // Trigger ParentAfterFork callbacks in parent process.
  ForkUtils::GetInstance().ParentAtFork();
}

void ChildAtFork() {
  MS_LOG(DEBUG) << "Child process at fork.";

  // Release the gil lock that was acquired manually before forking.
  if (!ForkUtils::GetInstance().IsGilHoldBeforeFork()) {
    MS_LOG(DEBUG) << "Release GIL lock acquired manually before fork.";
    PyGILState_Release(static_cast<PyGILState_STATE>(ForkUtils::GetInstance().GetGilState()));
  }

  // Trigger ChildAfterFork callbacks in child process.
  ForkUtils::GetInstance().ChildAtFork();
}

void SetPDeathSig(int signal) {
#ifdef __linux__
  // prctl(2) is a Linux specific system call.
  // On other systems the following function call has no effect.
  // This is set to ensure that non-daemonic child processes can
  // terminate if their parent terminates before they do.
  MS_LOG(DEBUG) << "Set prctl PR_SET_PDEATHSIG: " << signal;
  auto res = prctl(PR_SET_PDEATHSIG, signal);
  if (res < 0) {
    MS_LOG(WARNING) << "Set prctl PR_SET_PDEATHSIG failed:(" << errno << ")" << strerror(errno);
  }
#endif
}

void RegForkUtils(py::module *m) {
  auto m_sub = m->def_submodule("fork_utils", "submodule for fork");
  (void)m_sub.def("prepare_before_fork", &PrepareBeforeFork, "Callback function called in parent process before fork");
  (void)m_sub.def("parent_at_fork", &ParentAtFork, "Callback function called in parent process after fork");
  (void)m_sub.def("child_at_fork", &ChildAtFork, "Callback function called in child process after fork");
  (void)m_sub.def("prctl_set_pdeathsig", &SetPDeathSig, py::arg("signal"),
                  "Set signal to child process after parent process is dead");
}
}  // namespace mindspore
