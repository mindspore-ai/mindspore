/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/sig_handler.h"

#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/wait.h>
#include <sys/msg.h>
#include <sys/shm.h>
#endif

#include <csignal>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "minddata/dataset/util/task_manager.h"

namespace mindspore::dataset {
static std::unordered_map<int64_t, std::vector<int>> worker_groups = {};

// The shm_id & msg_id will be registered when the worker process run
// key: the process id
// value: the shm_id / msg_id
// Scenario 1: When the main process is killed, the worker process will release shm & msg in SIGTERMHandler
// Scenario 2: When the worker process is killed, the main process will release shm & msg in SIGCHLDHandler
std::mutex shm_msg_id_mtx_;                      // lock for g_shm_id & g_msg_id
static std::map<std::string, int32_t> g_shm_id;  // used by map / batch multiprocess mode data transfer
static std::map<std::string, int32_t> g_msg_id;  // used by map / batch multiprocess mode data transfer

#if !defined(_WIN32) && !defined(_WIN64)
/// \brief Set handler for the specified signal.
/// \param[in] signal The signal to set handler.
/// \param[in] handler The handler to execute when receive the signal.
/// \param[in] old_action The former handler.
void SetSignalHandler(int signal, void (*handler)(int, siginfo_t *, void *), struct sigaction *old_action) {
  struct sigaction action {};
  action.sa_sigaction = handler;
  action.sa_flags = SA_RESTART | SA_SIGINFO | SA_NOCLDSTOP | SA_NODEFER;
  if (sigemptyset(&action.sa_mask) != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to initialise the signal set, " << strerror(errno);
  }
  if (sigaction(signal, &action, old_action) != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to set handler for " << strsignal(signal) << ", " << strerror(errno);
  }
}

/// \brief A signal handler for SIGINT to interrupt watchdog.
/// \param[in] signal The signal that was raised.
/// \param[in] info The siginfo structure.
/// \param[in] context The context info.
void SIGINTHandler(int signal, siginfo_t *info, void *context) {
  // Wake up the watchdog which is designed as async-signal-safe.
  TaskManager::WakeUpWatchDog();
}

inline void OStreamWrite(int std_fileno, const char *error_msg) {
  if (!IS_OUTPUT_ON(mindspore::kInfo) && std_fileno == STDOUT_FILENO) {
    return;
  }
  auto ret = write(std_fileno, error_msg, strlen(error_msg));
  (void)ret;
}

// Memory cannot be dynamically allocated during the signal processing phase, so global variables are used here.
const char err_shmctl_delete[] = "shmctl delete shm_id failed.\n";
const char info_shmctl_delete[] = "Delete shared memory with shm_id successfully.\n";
const char err_msgctl_delelte[] = "msgctl delete msg_id failed.\n";
const char info_msgctl_delelte[] = "Delete message queue with msg_id successfully.\n";

void DoReleaseShmAndMsg(const std::string &key, bool need_lock = true) {
  // release the shm
  if (g_shm_id[key] != -1) {
    if (shmctl(g_shm_id[key], IPC_RMID, NULL) == -1 && errno != EINVAL) {
      // ignore the return value during the signal exit phase.
      OStreamWrite(STDERR_FILENO, err_shmctl_delete);
    } else {
      OStreamWrite(STDOUT_FILENO, info_shmctl_delete);
    }
  }

  // release the msg
  if (g_msg_id[key] != -1) {
    if (msgctl(g_msg_id[key], IPC_RMID, 0) == -1 && errno != EINVAL && errno != EIDRM) {
      OStreamWrite(STDERR_FILENO, err_msgctl_delelte);
    } else {
      OStreamWrite(STDOUT_FILENO, info_msgctl_delelte);
    }
  }

  if (need_lock) {
    std::lock_guard<std::mutex> lock(shm_msg_id_mtx_);
    g_shm_id[key] = -1;
    g_msg_id[key] = -1;
  } else {
    g_shm_id[key] = -1;
    g_msg_id[key] = -1;
  }
}

/// \brief Release the shared memory and message queue by process ids
/// \param[in] pids The map / batch worker process ids
void ReleaseShmAndMsgByWorkerPIDs(const std::vector<int> &pids) {
  std::string current_pid = std::to_string(getpid());
  for (auto &sub_pid : pids) {
    std::string key_prefix = current_pid + "_" + std::to_string(sub_pid);
    for (auto &item : g_shm_id) {
      if (item.first.find(key_prefix) != 0) {
        continue;
      }

      // release the shm and msg, it's run in main process thread
      // scenario 1: dataset.reset, when launch new worker processes, should close all old workers first which will
      //             release the shm queue and msg queue
      DoReleaseShmAndMsg(item.first, false);
    }
  }
  return;
}

// Memory cannot be dynamically allocated during the signal processing phase, so global variables are used here.
const int32_t g_pid_len = 64;
const int32_t DECIMAL_BASE = 10;
char g_current_pid[g_pid_len] = {0};
char g_ppid[g_pid_len] = {0};
char g_substr_ppid[g_pid_len] = {0};

const char message_memset[] = "memset_s failed.\n";
const char message_memcpy[] = "memcpy_s failed.\n";
const char message_pid_to_string[] = "pid to string failed.\n";
const char err_find_first[] = "Couldn't find first char '_' in the key.\n";
const char err_find_second[] = "Couldn't find second char '_' in the key.\n";
const char info1[] = "Get msg queue status failed.\n";
const char info2[] = "Get shm queue status failed.\n";
const char info3[] =
  "Parent process is still alive. And the msg & shm are used by current and parent process. "
  "No need to release the shm & msg by current process.\n";
const char info4[] =
  "Parent process is still alive. But the msg & shm is not used by parent process yet. "
  "Need to release the shm & msg by current process.\n";
const char info5[] = "Parent process is not alive. Need to release the shm & msg by current process.\n";

bool PIDToString(pid_t pid, char *buffer, size_t buffer_size) {
  int num = static_cast<int>(pid);
  char temp[g_pid_len] = {0};
  int i = 0;
  int j = 0;

  // num is 0
  if (num == 0) {
    if (buffer_size > 1) {
      buffer[0] = '0';
      buffer[1] = '\0';
      return true;
    }
    return false;
  }

  // extract number in reverse
  while (num > 0 && i < static_cast<int>(sizeof(temp)) - 1) {
    temp[i++] = static_cast<char>(static_cast<int>('0') + (num % DECIMAL_BASE));
    num /= DECIMAL_BASE;
  }

  // reverse the char
  while (--i >= 0 && j < static_cast<int>(buffer_size) - 1) {
    buffer[j++] = temp[i];
  }
  buffer[j] = '\0';
  return true;
}

// get the pid and ppid now
bool GetPIDAndPPID() {
  if (memset_s(g_current_pid, g_pid_len, 0, g_pid_len) != EOK) {
    // ignore the return value during the signal exit phase.
    OStreamWrite(STDERR_FILENO, message_memset);
    return false;
  }

  if (!PIDToString(getpid(), g_current_pid, g_pid_len)) {
    OStreamWrite(STDERR_FILENO, message_pid_to_string);
    return false;
  }

  if (memset_s(g_ppid, g_pid_len, 0, g_pid_len) != EOK) {
    OStreamWrite(STDERR_FILENO, message_memset);
    return false;
  }

  if (!PIDToString(getppid(), g_ppid, g_pid_len)) {
    OStreamWrite(STDERR_FILENO, message_pid_to_string);
    return false;
  }
  return true;
}

/// \brief Release the shared memory and message queue when got signal TERM / CHLD
void ReleaseShmAndMsg() {
  if (g_shm_id.empty()) {
    return;
  }

  if (!GetPIDAndPPID()) {
    return;
  }

  // release the shm & msg used by the current process when the main process is killed
  for (auto &item : g_shm_id) {
    // so just release the shm used by the current process
    // scenario 1: for the map / batch in process mode,
    //     the map / batch thread in main process, the item.first is MainProcessPID_WorkerPID_"PyFuncOp" /
    //                                                               MainProcessPID_WorkerPID_"BatchOp"
    //     the map / batch worker, the item.first is WorkerPID_MainProcessPID_"MapOp" /
    //                                               WorkerPID_MainProcessPID_"BatchOp"
    // scenario 2: for the independent dataset mode
    //     main process, the item.first is MainProcessPID_IndependentProcessPID_"ReceiveBridgeOp"
    //     independent process, the item.first is IndependentProcessPID_ParentPID_"SendBridgeOp"
    if (item.first.find(g_current_pid) != 0) {
      continue;
    }

    // no need to release the shm & msg in the map worker / batch worker / independent process when the main process
    // is still alive
    auto first_underline_char = item.first.find("_");
    if (first_underline_char == std::string::npos || first_underline_char <= 0) {
      // ignore the return value during the signal exit phase.
      OStreamWrite(STDERR_FILENO, err_find_first);
      return;
    }
    auto second_underline_char = item.first.find("_", first_underline_char + 1);
    if (second_underline_char == std::string::npos) {
      // ignore the return value during the signal exit phase.
      OStreamWrite(STDERR_FILENO, err_find_second);
      return;
    }

    if (memset_s(g_substr_ppid, g_pid_len, 0, g_pid_len) != EOK) {
      OStreamWrite(STDERR_FILENO, message_memset);
      return;
    }

    if (memcpy_s(g_substr_ppid, second_underline_char - first_underline_char - 1,
                 item.first.data() + first_underline_char + 1,
                 second_underline_char - first_underline_char - 1) != EOK) {
      OStreamWrite(STDERR_FILENO, message_memcpy);
      return;
    }

    // parent process is still alive, but the msg queue is not used
    // Scenario 1: when the independent dataset exit and the main process is still alive, not need to release shm & msg
    // Scenario 2: when the tree_adapter launch Python Workers success, but launch C++ op failed, the status.msg_stime
    //             is not changed. Should release the shm & msg
    if (strcmp(g_ppid, g_substr_ppid) == 0 && kill(std::stoi(g_ppid), 0) == 0) {
      // get the msg queue status
      msqid_ds msg_status;
      if (g_msg_id[item.first] != -1 && msgctl(g_msg_id[item.first], IPC_STAT, &msg_status) != 0) {
        // it may have already been released yet
        OStreamWrite(STDOUT_FILENO, info1);
      }

      // get the shm queue status
      shmid_ds shm_status;
      if (g_shm_id[item.first] != -1 && shmctl(g_shm_id[item.first], IPC_STAT, &shm_status) != 0) {
        // it may have already been released yet
        OStreamWrite(STDOUT_FILENO, info2);
      }

      // the msg & shm already be used by current process and parent process, it will be released by parent process
      if (msg_status.msg_stime != 0 && shm_status.shm_ctime != 0) {
        // Scenario 1
        OStreamWrite(STDOUT_FILENO, info3);
        continue;
      } else {  // the msg & shm just be used by current process, it will be released by current process
        // Scenario 2
        OStreamWrite(STDOUT_FILENO, info4);
      }
    } else {
      OStreamWrite(STDOUT_FILENO, info5);
    }

    // release the shm & msg
    DoReleaseShmAndMsg(item.first);
  }
}

const char err_sigterm[] = "[SIGTERMHandler] the signal is not SIGTERM.\n";
const char info6[] = "Dataset worker process was terminated by parent process, exits with successful status.\n";
const char err_sigemptyset[] = "Failed to initialise the signal set.\n";
const char err_sigaction[] = "Failed to set handler.\n";

/// \brief A signal handler for SIGTERM to exit the process.
/// \details When Python exits, it may terminate the children processes before deleting our runtime.
///   Then the watch dog has not been aborted, it will report an error and terminate the main process.
///   So we suppress SIGTERM sent from main process here by _exit(EXIT_SUCCESS).
/// \param[in] signal The signal that was raised.
/// \param[in] info The siginfo structure.
/// \param[in] context The context info.
void SIGTERMHandler(int signal, siginfo_t *info, void *context) {
  if (signal != SIGTERM) {
    OStreamWrite(STDERR_FILENO, err_sigterm);
    _exit(EXIT_FAILURE);
  }

  // release the shm & msg when the main process is killed
  ReleaseShmAndMsg();

  if (info->si_pid == getppid()) {
    OStreamWrite(STDOUT_FILENO, info6);
    _exit(EXIT_SUCCESS);
  }
  // reset the handler to the default
  struct sigaction term_action {};
  term_action.sa_handler = SIG_DFL;
  term_action.sa_flags = 0;
  if (sigemptyset(&term_action.sa_mask) != 0) {
    OStreamWrite(STDERR_FILENO, err_sigemptyset);
    _exit(EXIT_FAILURE);
  }
  if (sigaction(signal, &term_action, nullptr) != 0) {
    OStreamWrite(STDERR_FILENO, err_sigaction);
    _exit(EXIT_FAILURE);
  }
  raise(signal);
}

const char err_sigbus[] = "[SIGBUSHandler] the signal is not SIGBUS.\n";
const char err_bus_adrerr[] =
  "Unexpected bus error encountered in process. Non-existent physical address. "
  "This might be caused by insufficient shared memory. Please check if '/dev/shm' "
  "has enough available space via 'df -h'.\n";

/// \brief A signal handler for SIGBUS to retrieve the kill information.
/// \param[in] signal The signal that was raised.
/// \param[in] info The siginfo structure.
/// \param[in] context The context info.
void SIGBUSHandler(int signal, siginfo_t *info, void *context) {
  if (signal != SIGBUS) {
    OStreamWrite(STDERR_FILENO, err_sigbus);
    _exit(EXIT_FAILURE);
  }

  if (info->si_code == BUS_ADRERR) {
    OStreamWrite(STDERR_FILENO, err_bus_adrerr);
  }

  // reset the handler to the default
  struct sigaction bus_action {};
  bus_action.sa_handler = SIG_DFL;
  bus_action.sa_flags = 0;
  if (sigemptyset(&bus_action.sa_mask) != 0) {
    OStreamWrite(STDERR_FILENO, err_sigemptyset);
    _exit(EXIT_FAILURE);
  }
  if (sigaction(signal, &bus_action, nullptr) != 0) {
    OStreamWrite(STDERR_FILENO, err_sigaction);
    _exit(EXIT_FAILURE);
  }
  raise(signal);
}

/// \brief A signal handler for SIGSEGV to retrieve the segmentation information.
/// \param[in] signal The signal that was raised.
/// \param[in] info The siginfo structure.
/// \param[in] context The context info.
void SIGSEGVHandler(int signal, siginfo_t *info, void *context) {
  if (signal != SIGSEGV) {
    const char err_sigsegv[] = "[SIGSEGVHandler] the signal is not SIGSEGV.\n";
    OStreamWrite(STDERR_FILENO, err_sigsegv);
    _exit(EXIT_FAILURE);
  }

  if (info->si_code == SEGV_MAPERR) {
    const char segv_maperr[] = "Unexpected segmentation fault encountered in process. Address not mapped to object.\n";
    OStreamWrite(STDERR_FILENO, segv_maperr);
  } else if (info->si_code == SEGV_ACCERR) {
    const char segv_accerr[] =
      "Unexpected segmentation fault encountered in process. Invalid permissions for mapped object.\n";
    OStreamWrite(STDERR_FILENO, segv_accerr);
#ifdef SEGV_BNDERR
  } else if (info->si_code == SEGV_BNDERR) {
    const char segv_bnderr[] = "Unexpected segmentation fault encountered in process. Failed address bound checks.\n";
    OStreamWrite(STDERR_FILENO, segv_bnderr);
#endif
#ifdef SEGV_PKUERR
  } else if (info->si_code == SEGV_PKUERR) {
    const char segv_pkuerr[] =
      "Unexpected segmentation fault encountered in process. Access was denied by memory protection keys.\n";
    OStreamWrite(STDERR_FILENO, segv_pkuerr);
#endif
  } else {
    const char segv_err_normal[] = "Unexpected segmentation fault encountered in process.\n";
    OStreamWrite(STDERR_FILENO, segv_err_normal);
  }

  // reset the handler to the default
  struct sigaction segv_action {};
  segv_action.sa_handler = SIG_DFL;
  segv_action.sa_flags = 0;
  if (sigemptyset(&segv_action.sa_mask) != 0) {
    const char segv_err_sigemptyset[] = "Failed to initialise the signal set.\n";
    OStreamWrite(STDERR_FILENO, segv_err_sigemptyset);
    _exit(EXIT_FAILURE);
  }
  if (sigaction(signal, &segv_action, nullptr) != 0) {
    const char segv_err_sigaction[] = "Failed to set handler.\n";
    OStreamWrite(STDERR_FILENO, segv_err_sigaction);
    _exit(EXIT_FAILURE);
  }
  raise(signal);
}

/// \brief A signal handler for SIGFPE to retrieve the kill information.
/// \param[in] signal The signal that was raised.
/// \param[in] info The siginfo structure.
/// \param[in] context The context info.
void SIGFPEHandler(int signal, siginfo_t *info, void *context) {
  if (signal != SIGBUS) {
    const char err_sigfpe[] = "[SIGFPEHandler] the signal is not SIGFPE.\n";
    OStreamWrite(STDERR_FILENO, err_sigfpe);
    _exit(EXIT_FAILURE);
  }

  if (info->si_code == FPE_INTDIV) {
    const char sigfpe_err_intdiv[] =
      "Unexpected floating-point exception encountered in process. Integer divide by zero.\n";
    OStreamWrite(STDERR_FILENO, sigfpe_err_intdiv);
  } else if (info->si_code == FPE_INTOVF) {
    const char sigfpe_err_intovf[] = "Unexpected floating-point exception encountered in process. Integer overflow.\n";
    OStreamWrite(STDERR_FILENO, sigfpe_err_intovf);
  } else if (info->si_code == FPE_FLTDIV) {
    const char sigfpe_err_fltdiv[] =
      "Unexpected floating-point exception encountered in process. Floating-point divide by zero.\n";
    OStreamWrite(STDERR_FILENO, sigfpe_err_fltdiv);
  } else if (info->si_code == FPE_FLTOVF) {
    const char sigfpe_err_fltovf[] =
      "Unexpected floating-point exception encountered in process. Floating-point overflow.\n";
    OStreamWrite(STDERR_FILENO, sigfpe_err_fltovf);
  } else if (info->si_code == FPE_FLTUND) {
    const char sigfpe_err_fltund[] =
      "Unexpected floating-point exception encountered in process. Floating-point underflow.\n";
    OStreamWrite(STDERR_FILENO, sigfpe_err_fltund);
  } else if (info->si_code == FPE_FLTRES) {
    const char sigfpe_err_fltres[] =
      "Unexpected floating-point exception encountered in process. Floating-point inexact result.\n";
    OStreamWrite(STDERR_FILENO, sigfpe_err_fltres);
  } else if (info->si_code == FPE_FLTINV) {
    const char sigfpe_err_fltinv[] =
      "Unexpected floating-point exception encountered in process. Floating-point invalid operation.\n";
    OStreamWrite(STDERR_FILENO, sigfpe_err_fltinv);
  } else if (info->si_code == FPE_FLTSUB) {
    const char sigfpe_err_fltsub[] =
      "Unexpected floating-point exception encountered in process. Subscript out of range.\n";
    OStreamWrite(STDERR_FILENO, sigfpe_err_fltsub);
  } else {
    const char sigfpe_err_normal[] = "Unexpected floating-point exception encountered in process.\n";
    OStreamWrite(STDERR_FILENO, sigfpe_err_normal);
  }

  // reset the handler to the default
  struct sigaction fpe_action {};
  fpe_action.sa_handler = SIG_DFL;
  fpe_action.sa_flags = 0;
  if (sigemptyset(&fpe_action.sa_mask) != 0) {
    const char sigfpe_err_sigemptyset[] = "Failed to initialise the signal set.\n";
    OStreamWrite(STDERR_FILENO, sigfpe_err_sigemptyset);
    _exit(EXIT_FAILURE);
  }
  if (sigaction(signal, &fpe_action, nullptr) != 0) {
    const char sigfpe_err_sigaction[] = "Failed to set handler.\n";
    OStreamWrite(STDERR_FILENO, sigfpe_err_sigaction);
    _exit(EXIT_FAILURE);
  }
  raise(signal);
}

/// \brief A signal handler for SIGCHLD to clean the rest processes.
/// \param[in] signal The signal that was raised.
/// \param[in] info The siginfo structure.
/// \param[in] context The context info.
void SIGCHLDHandler(int signal, siginfo_t *info, void *context) {
  if (signal != SIGCHLD) {
    MS_LOG(ERROR) << "SIGCHLDHandler expects SIGCHLD signal, but got: " << strsignal(signal);
    _exit(EXIT_FAILURE);
  }

  for (auto &worker_group : worker_groups) {
    auto &pids = worker_group.second;
    int ppid = pids[0];
    if (getpid() != ppid) {
      continue;  // this worker group is not the children of current process
    }
    for (auto i = 1; i < pids.size(); ++i) {
      int pid = pids[i];
      siginfo_t sig_info{};
      sig_info.si_pid = 0;
      auto error = waitid(P_PID, pid, &sig_info, WEXITED | WNOHANG | WNOWAIT);
      std::string msg;
      if (error < 0) {
        if (errno == ECHILD) {
          msg = "Dataset worker process " + std::to_string(pid) +
                " has already exited. Its state may have been retrieved by other threads.";
        } else {
          MS_LOG(WARNING) << "Failed to wait for dataset worker process " << pid << ", got: " << strerror(errno);
          continue;
        }
      } else {
        if (sig_info.si_pid == 0) {
          continue;  // There were no children in a wait state.
        }
        if (sig_info.si_code == CLD_EXITED && sig_info.si_status != EXIT_SUCCESS) {  // exited unexpectedly
          msg = "Dataset worker process " + std::to_string(sig_info.si_pid) + " exited unexpectedly with exit code " +
                std::to_string(sig_info.si_status) + ".";
        } else if (sig_info.si_code == CLD_KILLED) {  // killed by signal
          msg = "Dataset worker process " + std::to_string(sig_info.si_pid) +
                " was killed by signal: " + std::string(strsignal(sig_info.si_status)) + ".";
        } else if (sig_info.si_code == CLD_DUMPED) {  // core dumped
          msg = "Dataset worker process " + std::to_string(sig_info.si_pid) +
                " core dumped: " + std::string(strsignal(sig_info.si_status)) + ".";
        } else {
          MS_LOG(INFO) << "Ignore dataset worker process " << pid << " with signal code " << sig_info.si_code;
          continue;
        }
      }

      // Start killing other child processes, ignoring the SIGCHLD signal to avoid being triggered repeatedly.
      // Scenario: kill -15 multi workers which may cause multi SIGCHLD
      ::signal(SIGCHLD, SIG_IGN);

      auto pids_to_kill = pids;
      pids.clear();  // Clear the monitoring status of the process group before performing a termination.
      for (const auto &pid_to_kill : pids_to_kill) {
        if (pid_to_kill != ppid && pid_to_kill != pid) {
          MS_LOG(INFO) << "Terminating child process: " << pid_to_kill;
          kill(pid_to_kill, SIGTERM);
        }
      }

      // release the shm & msg when the worker process is killed
      ReleaseShmAndMsg();

      MS_LOG(ERROR) << msg << " Main process will be terminated.";
      kill(getpid(), SIGTERM);
      // In case the signal is not responded, return here
      MS_LOG(WARNING) << "Main process may not respond to the SIGTERM signal, please check if it is blocked.";
      return;
    }
  }
}
#endif

std::string CheckIfWorkerExit() {
#if !defined(_WIN32) && !defined(_WIN64)
  for (auto &worker_group : worker_groups) {
    auto &pids = worker_group.second;
    int ppid = pids[0];
    if (getpid() != ppid) {
      continue;  // this worker group is not the children of current process
    }
    for (size_t i = 1; i < pids.size(); ++i) {
      int pid = pids[i];
      siginfo_t sig_info{};
      sig_info.si_pid = 0;
      auto error = waitid(P_PID, pid, &sig_info, WEXITED | WNOHANG | WNOWAIT);
      std::ostringstream out_stream;
      if (error < 0) {
        continue;
      }
      if (sig_info.si_pid == 0) {
        continue;  // There were no children in a wait state.
      }
      if (sig_info.si_code == CLD_EXITED && sig_info.si_status != EXIT_SUCCESS) {
        // exited unexpectedly
        out_stream << "DataLoader worker (pid: " << sig_info.si_pid << ") exited unexpectedly with exit code "
                   << sig_info.si_status << ".";
      } else if (sig_info.si_code == CLD_KILLED) {
        // killed by signal
        out_stream << "DataLoader worker (pid: " << sig_info.si_pid
                   << ") was killed by signal: " << strsignal(sig_info.si_status) << ".";
      } else if (sig_info.si_code == CLD_DUMPED) {
        // core dumped
        out_stream << "DataLoader worker (pid: " << sig_info.si_pid
                   << ") core dumped: " << strsignal(sig_info.si_status) << ".";
        if (sig_info.si_status == SIGBUS) {
          out_stream
            << " This might be caused by insufficient shared memory. Please check if '/dev/shm' has enough available "
               "space via 'df -h'.";
        }
      }
      pids.clear();  // Clear the monitoring status of the process group to avoid triggering this again.
      return out_stream.str();
    }
  }
#endif
  return "";
}

void RegisterHandlers() {
#if !defined(_WIN32) && !defined(_WIN64)
  SetSignalHandler(SIGINT, &SIGINTHandler, nullptr);
  SetSignalHandler(SIGTERM, &SIGINTHandler, nullptr);
#endif
}

void RegisterMainHandlers() {
#if !defined(_WIN32) && !defined(_WIN64)
  SetSignalHandler(SIGBUS, &SIGBUSHandler, nullptr);
  SetSignalHandler(SIGCHLD, &SIGCHLDHandler, nullptr);
#endif
}

void RegisterWorkerHandlers() {
#if !defined(_WIN32) && !defined(_WIN64)
  SetSignalHandler(SIGBUS, &SIGBUSHandler, nullptr);
  SetSignalHandler(SIGTERM, &SIGTERMHandler, nullptr);
  SetSignalHandler(SIGSEGV, &SIGSEGVHandler, nullptr);
  SetSignalHandler(SIGFPE, &SIGFPEHandler, nullptr);
#endif
}

std::string GetPIDsString(const std::vector<int> &pids) {
  std::string pids_string = "[";
  for (auto itr = pids.begin(); itr != pids.end(); ++itr) {
    if (itr != pids.begin()) {
      pids_string += ", ";
    }
    pids_string += std::to_string(*itr);
  }
  pids_string += "]";
  return pids_string;
}

void RegisterWorkerPIDs(int64_t id, const std::vector<int> &pids) {
  MS_LOG(INFO) << "Watch dog starts monitoring process(es): " << GetPIDsString(pids);
  worker_groups[id] = pids;
}

void DeregisterWorkerPIDs(int64_t id) {
  MS_LOG(INFO) << "Watch dog stops monitoring process(es): " << GetPIDsString(worker_groups[id]);
  (void)worker_groups.erase(id);
}

void RegisterShmIDAndMsgID(std::string pid, int32_t shm_id, int32_t msg_id) {
  {
    std::lock_guard<std::mutex> lock(shm_msg_id_mtx_);
    g_shm_id[pid] = shm_id;
    g_msg_id[pid] = msg_id;
  }
  MS_LOG(INFO) << "Update the shm_id to " << std::to_string(shm_id) << ", msg_id to " << std::to_string(msg_id)
               << " for pid: " << pid;
}

void UnlockShmIDAndMsgIDMutex() { shm_msg_id_mtx_.unlock(); }
}  // namespace mindspore::dataset
