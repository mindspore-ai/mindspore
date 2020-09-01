/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/engine/cache/cache_admin_arg.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <iostream>
#include <string>
#include <cstdlib>
#include "minddata/dataset/engine/cache/cache_request.h"
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/core/constants.h"

namespace mindspore {
namespace dataset {

const char CacheAdminArgHandler::kServerBinary[] = "cache_server";
const char CacheAdminArgHandler::kDefaultSpillDir[] = "/tmp";

CacheAdminArgHandler::CacheAdminArgHandler()
    : port_(kCfgDefaultCachePort),
      session_id_(0),
      num_workers_(kDefaultNumWorkers),
      shm_mem_sz_(kDefaultSharedMemorySizeInGB),
      log_level_(kDefaultLogLevel),
      hostname_(kCfgDefaultCacheHost),
      spill_dir_(kDefaultSpillDir),
      command_id_(CommandId::kCmdUnknown) {
  // Initialize the command mappings
  arg_map_["-h"] = ArgValue::kArgHost;
  arg_map_["--hostname"] = ArgValue::kArgHost;
  arg_map_["-p"] = ArgValue::kArgPort;
  arg_map_["--port"] = ArgValue::kArgPort;
  arg_map_["--start"] = ArgValue::kArgStart;
  arg_map_["--stop"] = ArgValue::kArgStop;
  arg_map_["--help"] = ArgValue::kArgHelp;
  arg_map_["--generate_session"] = ArgValue::kArgGenerateSession;
  arg_map_["-g"] = ArgValue::kArgGenerateSession;
  arg_map_["--destroy_session"] = ArgValue::kArgDestroySession;
  arg_map_["-d"] = ArgValue::kArgDestroySession;
  arg_map_["--spilldir"] = ArgValue::kArgSpillDir;
  arg_map_["-s"] = ArgValue::kArgSpillDir;
  arg_map_["-w"] = ArgValue::kArgNumWorkers;
  arg_map_["--workers"] = ArgValue::kArgNumWorkers;
  arg_map_["-m"] = ArgValue::kArgSharedMemorySize;
  arg_map_["--shared_memory_size"] = ArgValue::kArgSharedMemorySize;
  arg_map_["-l"] = ArgValue::kArgLogLevel;
  arg_map_["--minloglevel"] = ArgValue::kArgLogLevel;
  // Initialize argument tracker with false values
  for (int16_t i = 0; i < static_cast<int16_t>(ArgValue::kArgNumArgs); ++i) {
    ArgValue currAV = static_cast<ArgValue>(i);
    used_args_[currAV] = false;
  }
}

Status CacheAdminArgHandler::AssignArg(std::string option, int32_t *out_arg, std::stringstream *arg_stream,
                                       CommandId command_id) {
  // Detect if the user tried to provide this argument more than once
  ArgValue selected_arg = arg_map_[option];
  if (used_args_[selected_arg]) {
    std::string err_msg = "The " + option + " argument was given more than once.";
    return Status(StatusCode::kSyntaxError, err_msg);
  }

  // Flag that this arg is used now
  used_args_[selected_arg] = true;

  // Some options are just arguments, for example "--port 50052" is not a command, it's just a argument.
  // Other options are actual commands, for example "--destroy_session 1234".  This executes the destroy session.
  // If this option is also a command, make sure there has not been multiple commands given before assigning it.
  if (command_id != CommandId::kCmdUnknown) {
    if (command_id_ != CommandId::kCmdUnknown) {
      std::string err_msg = "Only one command at a time is allowed.  Invalid command: " + option;
      return Status(StatusCode::kSyntaxError, err_msg);
    } else {
      command_id_ = command_id;
    }
  }

  std::string value_as_string;

  // Fetch the argument from the arg stream into a string
  *arg_stream >> value_as_string;
  if (value_as_string.empty()) {
    std::string err_msg = option + " option requires an argument field.  Syntax: " + option + " <field>";
    return Status(StatusCode::kSyntaxError, err_msg);
  }

  // Now, attempt to convert the value into it's string format for output
  try {
    *out_arg = std::stoul(value_as_string);
  } catch (const std::exception &e) {
    std::string err_msg = "Invalid numeric value: " + value_as_string;
    return Status(StatusCode::kSyntaxError, err_msg);
  }

  return Status::OK();
}

Status CacheAdminArgHandler::AssignArg(std::string option, std::string *out_arg, std::stringstream *arg_stream,
                                       CommandId command_id) {
  // Detect if the user tried to provide this argument more than once
  ArgValue selected_arg = arg_map_[option];
  if (used_args_[selected_arg]) {
    std::string err_msg = "The " + option + " argument was given more than once.";
    return Status(StatusCode::kSyntaxError, err_msg);
  }

  // Flag that this arg is used now
  used_args_[selected_arg] = true;

  // Some options are just arguments, for example "--hostname "127.0.0.1" is not a command, it's just an argument.
  // Other options are actual commands, for example "--start".
  // If this option is also a command, make sure there has not been multiple commands given before assigning it.
  if (command_id != CommandId::kCmdUnknown) {
    if (command_id_ != CommandId::kCmdUnknown) {
      std::string err_msg = "Only one command at a time is allowed.  Invalid command: " + option;
      return Status(StatusCode::kSyntaxError, err_msg);
    } else {
      command_id_ = command_id;
    }
  }

  // If there is no argument to get, such as the --start command, then out_arg will be a nullptr.
  if (out_arg != nullptr) {
    // Fetch the argument from the arg stream into a string
    *arg_stream >> *out_arg;
    if (out_arg->empty()) {
      std::string err_msg = option + " option requires an argument field.  Syntax: " + option + " <field>";
      return Status(StatusCode::kSyntaxError, err_msg);
    }
  }

  return Status::OK();
}

Status CacheAdminArgHandler::ParseArgStream(std::stringstream *arg_stream) {
  std::string tok;
  while (*arg_stream >> tok) {
    switch (arg_map_[tok]) {
      case ArgValue::kArgHost: {
        RETURN_IF_NOT_OK(AssignArg(tok, &hostname_, arg_stream));
        break;
      }
      case ArgValue::kArgPort: {
        RETURN_IF_NOT_OK(AssignArg(tok, &port_, arg_stream));
        break;
      }
      case ArgValue::kArgStart: {
        RETURN_IF_NOT_OK(AssignArg(tok, static_cast<std::string *>(nullptr), arg_stream, CommandId::kCmdStart));
        break;
      }
      case ArgValue::kArgStop: {
        RETURN_IF_NOT_OK(AssignArg(tok, static_cast<std::string *>(nullptr), arg_stream, CommandId::kCmdStop));
        break;
      }
      case ArgValue::kArgGenerateSession: {
        RETURN_IF_NOT_OK(
          AssignArg(tok, static_cast<std::string *>(nullptr), arg_stream, CommandId::kCmdGenerateSession));
        break;
      }
      case ArgValue::kArgHelp: {
        command_id_ = CommandId::kCmdHelp;
        break;
      }
      case ArgValue::kArgDestroySession: {
        // session_id is an unsigned type. We may need to template the AssignArg function so that
        // it can handle different flavours of integers instead of just int32_t.
        int32_t session_int;
        RETURN_IF_NOT_OK(AssignArg(tok, &session_int, arg_stream, CommandId::kCmdDestroySession));
        session_id_ = session_int;
        break;
      }
      case ArgValue::kArgNumWorkers: {
        RETURN_IF_NOT_OK(AssignArg(tok, &num_workers_, arg_stream));
        break;
      }
      case ArgValue::kArgSpillDir: {
        RETURN_IF_NOT_OK(AssignArg(tok, &spill_dir_, arg_stream));
        break;
      }
      case ArgValue::kArgSharedMemorySize: {
        RETURN_IF_NOT_OK(AssignArg(tok, &shm_mem_sz_, arg_stream));
        break;
      }
      case ArgValue::kArgLogLevel: {
        RETURN_IF_NOT_OK(AssignArg(tok, &log_level_, arg_stream));
        break;
      }
      default: {
        // Save space delimited trailing arguments
        trailing_args_ += (" " + tok);
        break;
      }
    }
  }

  RETURN_IF_NOT_OK(Validate());

  return Status::OK();
}

Status CacheAdminArgHandler::Validate() {
  // This sanity check is delayed until now in case there may be valid use-cases of trailing args.
  // Any unhandled arguments at this point is an error.
  if (!trailing_args_.empty()) {
    std::string err_msg = "Invalid arguments provided: " + trailing_args_;
    return Status(StatusCode::kSyntaxError, err_msg);
  }

  // The user must pick at least one command.  i.e. it's meaningless to just give a hostname or port but no command to
  // run.
  if (command_id_ == CommandId::kCmdUnknown) {
    std::string err_msg = "No command provided";
    return Status(StatusCode::kSyntaxError, err_msg);
  }

  // Additional checks here
  if (num_workers_ < 1) return Status(StatusCode::kSyntaxError, "Number of workers must be positive value.");
  if (log_level_ < 0 || log_level_ > 3) return Status(StatusCode::kSyntaxError, "Log level must be in range (0..3).");
  // port range check?

  return Status::OK();
}

Status CacheAdminArgHandler::RunCommand() {
  switch (command_id_) {
    case CommandId::kCmdHelp: {
      Help();
      break;
    }
    case CommandId::kCmdStart: {
      RETURN_IF_NOT_OK(StartServer());
      break;
    }
    case CommandId::kCmdStop: {
      RETURN_IF_NOT_OK(StopServer());
      break;
    }
    case CommandId::kCmdGenerateSession: {
      CacheClientGreeter comm(hostname_, port_, 1);
      RETURN_IF_NOT_OK(comm.ServiceStart());
      auto rq = std::make_shared<GenerateSessionIdRequest>();
      RETURN_IF_NOT_OK(comm.HandleRequest(rq));
      RETURN_IF_NOT_OK(rq->Wait());
      std::cout << rq->GetSessionId() << std::endl;
      break;
    }
    case CommandId::kCmdDestroySession: {
      CacheClientGreeter comm(hostname_, port_, 1);
      RETURN_IF_NOT_OK(comm.ServiceStart());
      CacheClientInfo cinfo;
      cinfo.set_session_id(session_id_);
      auto rq = std::make_shared<DropSessionRequest>(cinfo);
      RETURN_IF_NOT_OK(comm.HandleRequest(rq));
      RETURN_IF_NOT_OK(rq->Wait());
      std::cout << "Drop session successful" << std::endl;
      break;
    }
    default: {
      RETURN_STATUS_UNEXPECTED("Invalid cache admin command id.");
      break;
    }
  }

  return Status::OK();
}

Status CacheAdminArgHandler::StartServer() {
  // There currently does not exist any "install path" or method to identify which path the installed binaries will
  // exist in. As a temporary approach, we will assume that the server binary shall exist in the same path as the
  // cache_admin binary (this process).
  const std::string self_proc = "/proc/self/exe";
  std::string canonical_path;
  canonical_path.resize(400);  // PATH_MAX is large. This value should be big enough for our use.
  // Some lower level OS library calls are needed here to determine the binary path.
  // Fetch the path of this binary for admin_cache into C character array and then truncate off the binary name so that
  // we are left with only the absolute path
  if (realpath(self_proc.data(), canonical_path.data()) == nullptr) {
    std::string err_msg = "Failed to identify cache admin binary path: " + std::to_string(errno);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  canonical_path.resize(strlen(canonical_path.data()));
  int last_seperator = canonical_path.find_last_of('/');
  CHECK_FAIL_RETURN_UNEXPECTED(last_seperator != std::string::npos, "No / found");
  // truncate the binary name so we are left with the absolute path of cache_admin binary
  canonical_path.resize(last_seperator + 1);
  std::string cache_server_binary = canonical_path + std::string(kServerBinary);

  // Create a pipe before we fork. If all goes well, the child will run as a daemon in the background
  // and never returns until shutdown. If there is any error, the child will notify us through the pipe.
  int fd[2];
  if (pipe(fd) == -1) {
    std::string err_msg = "Failed to create a pipe for communication " + std::to_string(errno);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // fork the child process to become the daemon
  pid_t pid;
  pid = fork();

  // failed to fork
  if (pid < 0) {
    std::string err_msg = "Failed to fork process for cache server: " + std::to_string(errno);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else if (pid > 0) {
    // As a parent, we close the write end. We only listen.
    close(fd[1]);
    dup2(fd[0], 0);
    close(fd[0]);
    wait(nullptr);
    std::string msg;
    const int32_t buf_sz = 1024;
    msg.resize(buf_sz);
    auto n = read(0, msg.data(), buf_sz);
    if (n < 0) {
      std::string err_msg = "Failed to read from pipeline " + std::to_string(errno);
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    msg.resize(n);
    std::cout << msg << std::endl;
    return Status::OK();
  } else {
    // Child here ...
    // Close all stdin, redirect stdout and stderr to the write end of the pipe.
    close(fd[0]);
    dup2(fd[1], 1);
    dup2(fd[1], 2);
    close(0);
    close(fd[1]);
    // exec the cache server binary in this process
    std::string port_string = std::to_string(port_);
    std::string workers_string = std::to_string(num_workers_);
    std::string shared_memory_string = std::to_string(shm_mem_sz_);
    std::string minloglevel_string = std::to_string(log_level_);
    std::string daemonize_string = "true";

    char *argv[8];
    argv[0] = cache_server_binary.data();  // First arg is usually the binary name
    argv[1] = spill_dir_.data();
    argv[2] = workers_string.data();
    argv[3] = port_string.data();
    argv[4] = shared_memory_string.data();
    argv[5] = minloglevel_string.data();
    argv[6] = daemonize_string.data();
    argv[7] = nullptr;

    // Now exec the binary
    execv(argv[0], argv);
    // If the exec was successful, this line will never be reached due to process image being replaced.
    // ..unless exec failed.
    std::string err_msg = "Failed to exec cache server: " + cache_server_binary;
    std::cerr << err_msg << std::endl;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
}

Status CacheAdminArgHandler::StopServer() {
  CacheClientGreeter comm(hostname_, port_, 1);
  RETURN_IF_NOT_OK(comm.ServiceStart());
  auto rq = std::make_shared<ShutdownRequest>();
  RETURN_IF_NOT_OK(comm.HandleRequest(rq));
  // We will ignore the rc because if the shutdown is successful, the server will not reply back.
  (void)rq->Wait();
  return Status::OK();
}

void CacheAdminArgHandler::Help() {
  std::cerr << "Syntax:\n";
  std::cerr << "   cache_admin [--start | --stop]\n";
  std::cerr << "               [ [-h | --hostname] <hostname> ]\n";
  std::cerr << "               [ [-p | --port] <port number> ]\n";
  std::cerr << "               [ [-g | --generate_session] ]\n";
  std::cerr << "               [ [-d | --destroy_session] <session id> ]\n";
  std::cerr << "               [ [-w | --workers] <number of workers> ]\n";
  std::cerr << "               [ [-s | --spilldir] <spilling directory> ]\n";
  std::cerr << "               [ [-m | --shared_memory_size] <shared memory size> ]\n";
  std::cerr << "               [ [-l | --minloglevel] <log level> ]\n";
  std::cerr << "               [--help]" << std::endl;
}
}  // namespace dataset
}  // namespace mindspore
