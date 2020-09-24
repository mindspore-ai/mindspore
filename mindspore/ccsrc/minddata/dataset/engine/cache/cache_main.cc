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

#include "minddata/dataset/engine/cache/cache_server.h"
#include <sys/types.h>
#include <unistd.h>
#ifdef USE_GLOG
#include <glog/logging.h>
#endif
#include <cstdlib>
#include <thread>
#include <chrono>
#include "minddata/dataset/engine/cache/cache_common.h"
#include "minddata/dataset/engine/cache/cache_ipc.h"
namespace ds = mindspore::dataset;

/// Send a synchronous command to the local server using tcp/ip.
/// We aren't using any client code because this binary is not necessarily linked with the client library.
/// So just using grpc call directly.
/// \param port tcp/ip port to use
/// \param type Type of command.
/// \param out grpc result
/// \return Status object
ds::Status SendSyncCommand(int32_t port, ds::BaseRequest::RequestType type, ds::CacheRequest *rq, ds::CacheReply *reply,
                           grpc::Status *out) {
  if (rq == nullptr) {
    return ds::Status(ds::StatusCode::kUnexpectedError, "pointer rq is null");
  }
  if (reply == nullptr) {
    return ds::Status(ds::StatusCode::kUnexpectedError, "pointer reply is null");
  }
  if (out == nullptr) {
    return ds::Status(ds::StatusCode::kUnexpectedError, "pointer out is null");
  }
  const std::string hostname = "127.0.0.1";
  auto unix_socket = ds::PortToUnixSocketPath(port);
#if CACHE_LOCAL_CLIENT
  const std::string target = "unix://" + unix_socket;
#else
  const std::string target = hostname + ":" + std::to_string(port);
#endif
  try {
    rq->set_type(static_cast<int16_t>(type));
    grpc::ChannelArguments args;
    grpc::ClientContext ctx;
    grpc::CompletionQueue cq;
    // Standard async rpc call
    std::shared_ptr<grpc::Channel> channel =
      grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), args);
    std::unique_ptr<ds::CacheServerGreeter::Stub> stub = ds::CacheServerGreeter::NewStub(channel);
    std::unique_ptr<grpc::ClientAsyncResponseReader<ds::CacheReply>> rpc =
      stub->PrepareAsyncCacheServerRequest(&ctx, *rq, &cq);
    rpc->StartCall();
    // We need to pass a tag. But since this is the only request in the completion queue and so we
    // just pass a dummy
    int64_t dummy;
    void *tag;
    bool success;
    rpc->Finish(reply, out, &dummy);
    // Now we wait on the completion queue synchronously.
    auto r = cq.Next(&tag, &success);
    if (r == grpc_impl::CompletionQueue::NextStatus::GOT_EVENT) {
      if (!success || tag != &dummy) {
        std::string errMsg = "Unexpected programming error ";
        return ds::Status(ds::StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
      }
      if (out->ok()) {
        return ds::Status(static_cast<ds::StatusCode>(reply->rc()), reply->msg());
      } else {
        auto error_code = out->error_code();
        std::string errMsg = out->error_message() + ". GRPC Code " + std::to_string(error_code);
        return ds::Status(ds::StatusCode::kNetWorkError, errMsg);
      }
    } else {
      std::string errMsg = "Unexpected queue rc = " + std::to_string(r);
      return ds::Status(ds::StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
    }
  } catch (const std::exception &e) {
    return ds::Status(ds::StatusCode::kUnexpectedError, __LINE__, __FILE__, e.what());
  }
}

/// Stop the server
/// \param argv
/// \return Status object
ds::Status StopServer(int argc, char **argv) {
  ds::Status rc;
  ds::CacheServer::Builder builder;
  std::string errMsg;
  if (argc != 2) {
    return ds::Status(ds::StatusCode::kSyntaxError);
  }
  int32_t port = strtol(argv[1], nullptr, 10);
  // We will go through the builder to do some snaity check. We only need the port number
  // to shut down the server. Null the root directory as we don't trigger the sanity code to write out anything
  // to the spill directory.
  builder.SetPort(port).SetRootDirectory("");
  // Part of the sanity check is check the shared memory. If the server is up and running, we expect
  // the return code is kDuplicate.
  rc = builder.SanityCheck();
  if (rc.IsOk()) {
    errMsg = "Server is not up or has been shutdown already.";
    return ds::Status(ds::StatusCode::kUnexpectedError, errMsg);
  } else if (rc.get_code() != ds::StatusCode::kDuplicateKey) {
    // Not OK, and no duplicate, just return the rc whatever it is.
    return rc;
  } else {
    // Now we get some work to do. We will send a tcp/ip request to the given port.
    // This binary is not linked with client side of code, so we will just call grpc directly.
    ds::CacheRequest rq;
    ds::CacheReply reply;
    grpc::Status grpc_rc;
    rc = SendSyncCommand(port, ds::BaseRequest::RequestType::kStopService, &rq, &reply, &grpc_rc);
    // The request is like a self destruct message, the server will not send anything back and
    // shutdown all incoming request. So we should expect some unexpected network error if
    // all goes well and we expect to GRPC code 14.
    auto err_code = grpc_rc.error_code();
    if (rc.get_code() != ds::StatusCode::kNetWorkError || err_code != grpc::StatusCode::UNAVAILABLE) {
      return ds::Status(ds::StatusCode::kUnexpectedError, __LINE__, __FILE__);
    }
  }
  return ds::Status::OK();
}

/// Start the server
/// \param argv
/// \return Status object
ds::Status StartServer(int argc, char **argv) {
  ds::Status rc;
  ds::CacheServer::Builder builder;
  if (argc != 8) {
    return ds::Status(ds::StatusCode::kSyntaxError);
  }

  int32_t port = strtol(argv[3], nullptr, 10);
  builder.SetRootDirectory(argv[1])
    .SetNumWorkers(strtol(argv[2], nullptr, 10))
    .SetPort(port)
    .SetSharedMemorySizeInGB(strtol(argv[4], nullptr, 10))
    .SetMemoryCapRatio(strtof(argv[7], nullptr));

#ifdef USE_GLOG
  FLAGS_minloglevel = strtol(argv[5], nullptr, 10);
#endif

  auto daemonize_string = argv[6];
  bool daemonize = strcmp(daemonize_string, "true") == 0 || strcmp(daemonize_string, "TRUE") == 0 ||
                   strcmp(daemonize_string, "t") == 0 || strcmp(daemonize_string, "T") == 0;

  // We always change directory to / on unix rather than using the directory where the cache_server
  // is called. This is a standard procedure for daemonize a process on unix.
  if (chdir("/") == -1) {
    std::string errMsg = "Unable to change directory to /. Errno = " + std::to_string(errno);
    return ds::Status(ds::StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  }

  // A message queue for communication between parent and child (if we fork).
  ds::SharedMessage msg;
  if (daemonize) {
#ifdef USE_GLOG
    FLAGS_log_dir = "/tmp";
    google::InitGoogleLogging(argv[0]);
#endif
    rc = msg.Create();
    if (rc.IsError()) {
      return rc;
    }
    pid_t pid = fork();
    // failed to fork
    if (pid < 0) {
      std::string errMsg = "Failed to fork process for cache server. Errno = " + std::to_string(errno);
      return ds::Status(ds::StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
    } else if (pid > 0) {
      // Parent and will be responsible for remove the queue on exit.
      msg.RemoveResourcesOnExit();
      // Sleep one second and we attach to the msg que
      std::this_thread::sleep_for(std::chrono::seconds(1));
      ds::Status child_rc;
      rc = msg.ReceiveStatus(&child_rc);
      if (rc.IsError()) {
        return rc;
      }
      if (child_rc.IsError()) {
        return child_rc;
      }
      std::cerr << "cache server daemon process has been created as process id: " << pid
                << "\nCheck log file for any start up error" << std::endl;
      signal(SIGCHLD, SIG_IGN);  // ignore sig child signal.
      return ds::Status::OK();
    } else {
      // Child process will continue from here if daemonize and parent has already exited.
      // If we are running in the foreground, none of the code in block below will be run.
      pid_t sid;
      umask(0);
      sid = setsid();
      if (sid < 0) {
        std::string errMsg = "Failed to setsid(). Errno = " + std::to_string(errno);
        return ds::Status(ds::StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
      }
      close(0);
      close(1);
      close(2);
    }
  }

  // Dump the summary
  MS_LOG(INFO) << builder << std::endl;
  // Create the instance with some sanity checks built in
  rc = builder.Build();
  if (rc.IsOk()) {
    // If all goes well, kick off the threads. Loop forever and never return unless error.
    ds::CacheServer &cs = ds::CacheServer::GetInstance();
    rc = cs.Run(msg.GetMsgQueueId());
  } else if (daemonize) {
    // If we didn't pass the sanity check to at least create the instance, use
    // the message queue to return the error message if this is the child daemon.
    return msg.SendStatus(rc);
  }
  return rc;
}

int main(int argc, char **argv) {
  ds::Status rc;
  ds::CacheServer::Builder builder;

  // This executable is not to be called directly, and should be invoked by cache_admin executable.
  if (strcmp(argv[0], "-") == 0) {
    rc = StopServer(argc, argv);
  } else {
    rc = StartServer(argc, argv);
  }
  // Check result
  if (rc.IsError()) {
    auto errCode = rc.get_code();
    auto errMsg = rc.ToString();
    std::cerr << errMsg << std::endl;
    return static_cast<int>(errCode);
  }
  return 0;
}
