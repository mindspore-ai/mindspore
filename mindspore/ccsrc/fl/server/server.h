/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_SERVER_H_
#define MINDSPORE_CCSRC_FL_SERVER_SERVER_H_

#include <memory>
#include <string>
#include <vector>
#include "ps/core/communicator/communicator_base.h"
#include "ps/core/communicator/tcp_communicator.h"
#include "ps/core/communicator/task_executor.h"
#include "ps/core/file_configuration.h"
#include "fl/server/common.h"
#include "fl/server/executor.h"
#include "fl/server/iteration.h"
#ifdef ENABLE_ARMOUR
#include "fl/armour/cipher/cipher_init.h"
#endif

namespace mindspore {
namespace fl {
namespace server {
// The sleeping time of the server thread before the networking is completed.
constexpr uint32_t kServerSleepTimeForNetworking = 1000;

// Class Server is the entrance of MindSpore's parameter server training mode and federated learning.
class Server {
 public:
  static Server &GetInstance() {
    static Server instance;
    return instance;
  }

  void Initialize(bool use_tcp, bool use_http, uint16_t http_port, const std::vector<RoundConfig> &rounds_config,
                  const CipherConfig &cipher_config, const FuncGraphPtr &func_graph, size_t executor_threshold);

  // According to the current MindSpore framework, method Run is a step of the server pipeline. This method will be
  // blocked until the server is finalized.
  // func_graph is the frontend graph which will be parse in server's exector and aggregator.
  void Run();

  void SwitchToSafeMode();
  void CancelSafeMode();
  bool IsSafeMode() const;
  void WaitExitSafeMode() const;

  // Whether the training job of the server is enabled.
  InstanceState instance_state() const;

 private:
  Server()
      : server_node_(nullptr),
        task_executor_(nullptr),
        use_tcp_(false),
        use_http_(false),
        http_port_(0),
        func_graph_(nullptr),
        executor_threshold_(0),
        communicator_with_server_(nullptr),
        communicators_with_worker_({}),
        iteration_(nullptr),
        safemode_(true),
        scheduler_ip_(""),
        scheduler_port_(0),
        server_num_(0),
        worker_num_(0),
        fl_server_port_(0),
        cipher_initial_client_cnt_(0),
        cipher_exchange_keys_cnt_(0),
        cipher_get_keys_cnt_(0),
        cipher_share_secrets_cnt_(0),
        cipher_get_secrets_cnt_(0),
        cipher_get_clientlist_cnt_(0),
        cipher_reconstruct_secrets_up_cnt_(0),
        cipher_reconstruct_secrets_down_cnt_(0),
        cipher_time_window_(0) {}
  ~Server() = default;
  Server(const Server &) = delete;
  Server &operator=(const Server &) = delete;

  // Load variables which is set by ps_context.
  void InitServerContext();

  // Try to recover server config from persistent storage.
  void Recovery();

  // Initialize the server cluster, server node and communicators.
  void InitCluster();
  bool InitCommunicatorWithServer();
  bool InitCommunicatorWithWorker();

  // Initialize iteration with rounds. Which rounds to use could be set by ps_context as well.
  void InitIteration();

  // Register all message and event callbacks for communicators(TCP and HTTP). This method must be called before
  // communicators are started.
  void RegisterCommCallbacks();

  // Register cluster exception callbacks. This method is called in RegisterCommCallbacks.
  void RegisterExceptionEventCallback(const std::shared_ptr<ps::core::TcpCommunicator> &communicator);

  // Register message callbacks. These messages are mainly from scheduler.
  void RegisterMessageCallback(const std::shared_ptr<ps::core::TcpCommunicator> &communicator);

  // Initialize executor according to the server mode.
  void InitExecutor();

  // Initialize cipher according to the public param.
  void InitCipher();

  // Create round kernels and bind these kernels with corresponding Round.
  void RegisterRoundKernel();

  void InitMetrics();

  // The communicators should be started after all initializations are completed.
  void StartCommunicator();

  // The barriers before scaling operations.
  void ProcessBeforeScalingOut();
  void ProcessBeforeScalingIn();

  // The handlers after scheduler's scaling operations are done.
  void ProcessAfterScalingOut();
  void ProcessAfterScalingIn();

  // Handlers for enableFLS/disableFLS requests from the scheduler.
  void HandleEnableServerRequest(const std::shared_ptr<ps::core::MessageHandler> &message);
  void HandleDisableServerRequest(const std::shared_ptr<ps::core::MessageHandler> &message);

  // Finish current instance and start a new one. FLPlan could be changed in this method.
  void HandleNewInstanceRequest(const std::shared_ptr<ps::core::MessageHandler> &message);

  // Query current instance information.
  void HandleQueryInstanceRequest(const std::shared_ptr<ps::core::MessageHandler> &message);

  // The server node is initialized in Server.
  std::shared_ptr<ps::core::ServerNode> server_node_;

  // The task executor of the communicators. This helps server to handle network message concurrently. The tasks
  // submitted to this task executor is asynchronous.
  std::shared_ptr<ps::core::TaskExecutor> task_executor_;

  // Which protocol should communicators use.
  bool use_tcp_;
  bool use_http_;
  uint16_t http_port_;

  // The configure of all rounds.
  std::vector<RoundConfig> rounds_config_;
  CipherConfig cipher_config_;

  // The graph passed by the frontend without backend optimizing.
  FuncGraphPtr func_graph_;

  // The threshold count for executor to do aggregation or optimizing.
  size_t executor_threshold_;

  // Server need a tcp communicator to communicate with other servers for counting, metadata storing, collective
  // operations, etc.
  std::shared_ptr<ps::core::CommunicatorBase> communicator_with_server_;

  // The communication with workers(including mobile devices), has multiple protocol types: HTTP and TCP.
  // In some cases, both types should be supported in one distributed training job. So here we may have multiple
  // communicators.
  std::vector<std::shared_ptr<ps::core::CommunicatorBase>> communicators_with_worker_;

  // Mutex for scaling operations. We must wait server's initialization done before handle scaling events.
  std::mutex scaling_mtx_;

  // Iteration consists of multiple kinds of rounds.
  Iteration *iteration_;

  // The flag that represents whether server is in safemode.
  // If true, the server is not available to workers and clients.
  std::atomic_bool safemode_;

  // Variables set by ps context.
#ifdef ENABLE_ARMOUR
  armour::CipherInit *cipher_init_{nullptr};
#endif
  std::string scheduler_ip_;
  uint16_t scheduler_port_;
  uint32_t server_num_;
  uint32_t worker_num_;
  uint16_t fl_server_port_;
  size_t cipher_initial_client_cnt_;
  size_t cipher_exchange_keys_cnt_;
  size_t cipher_get_keys_cnt_;
  size_t cipher_share_secrets_cnt_;
  size_t cipher_get_secrets_cnt_;
  size_t cipher_get_clientlist_cnt_;
  size_t cipher_reconstruct_secrets_up_cnt_;
  size_t cipher_reconstruct_secrets_down_cnt_;
  uint64_t cipher_time_window_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_SERVER_H_
