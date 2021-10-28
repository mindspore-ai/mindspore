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

#include "fl/server/server.h"
#include <memory>
#include <string>
#include <csignal>
#ifdef ENABLE_ARMOUR
#include "fl/armour/secure_protocol/secret_sharing.h"
#endif
#include "fl/server/round.h"
#include "fl/server/model_store.h"
#include "fl/server/iteration.h"
#include "fl/server/collective_ops_impl.h"
#include "fl/server/distributed_metadata_store.h"
#include "fl/server/distributed_count_service.h"
#include "fl/server/kernel/round/round_kernel_factory.h"

namespace mindspore {
namespace fl {
namespace server {
// The handler to capture the signal of SIGTERM. Normally this signal is triggered by cloud cluster managers like K8S.
std::shared_ptr<ps::core::CommunicatorBase> g_communicator_with_server = nullptr;
std::vector<std::shared_ptr<ps::core::CommunicatorBase>> g_communicators_with_worker = {};
void SignalHandler(int signal) {
  MS_LOG(WARNING) << "SIGTERM captured: " << signal;
  (void)std::for_each(g_communicators_with_worker.begin(), g_communicators_with_worker.end(),
                      [](const std::shared_ptr<ps::core::CommunicatorBase> &communicator) {
                        MS_ERROR_IF_NULL_WO_RET_VAL(communicator);
                        (void)communicator->Stop();
                      });

  MS_ERROR_IF_NULL_WO_RET_VAL(g_communicator_with_server);
  (void)g_communicator_with_server->Stop();
  return;
}

void Server::Initialize(bool use_tcp, bool use_http, uint16_t http_port, const std::vector<RoundConfig> &rounds_config,
                        const CipherConfig &cipher_config, const FuncGraphPtr &func_graph, size_t executor_threshold) {
  MS_EXCEPTION_IF_NULL(func_graph);
  func_graph_ = func_graph;

  if (rounds_config.empty()) {
    MS_LOG(EXCEPTION) << "Rounds are empty.";
    return;
  }
  rounds_config_ = rounds_config;
  cipher_config_ = cipher_config;

  use_tcp_ = use_tcp;
  use_http_ = use_http;
  http_port_ = http_port;
  executor_threshold_ = executor_threshold;
  (void)signal(SIGTERM, SignalHandler);
  return;
}

// Each step of the server pipeline may have dependency on other steps, which includes:

// InitServerContext must be the first step to set contexts for later steps.

// Server Running relies on URL or Message Type Register:
// StartCommunicator---->InitIteration

// Metadata Register relies on Hash Ring of Servers which relies on Network Building Completion:
// RegisterRoundKernel---->StartCommunicator

// Kernel Initialization relies on Executor Initialization:
// RegisterRoundKernel---->InitExecutor

// Getting Model Size relies on ModelStorage Initialization which relies on Executor Initialization:
// InitCipher---->InitExecutor
void Server::Run() {
  std::unique_lock<std::mutex> lock(scaling_mtx_);
  InitServerContext();
  InitCluster();
  InitIteration();
  RegisterCommCallbacks();
  StartCommunicator();
  InitExecutor();
  std::string encrypt_type = ps::PSContext::instance()->encrypt_type();
  if (encrypt_type != ps::kNotEncryptType) {
    InitCipher();
    MS_LOG(INFO) << "Parameters for secure aggregation have been initiated.";
  }
  RegisterRoundKernel();
  InitMetrics();
  MS_LOG(INFO) << "Server started successfully.";
  safemode_ = false;
  lock.unlock();

  // Wait communicators to stop so the main thread is blocked.
  (void)std::for_each(communicators_with_worker_.begin(), communicators_with_worker_.end(),
                      [](const std::shared_ptr<ps::core::CommunicatorBase> &communicator) {
                        MS_EXCEPTION_IF_NULL(communicator);
                        communicator->Join();
                      });
  MS_EXCEPTION_IF_NULL(communicator_with_server_);
  communicator_with_server_->Join();
  MsException::Instance().CheckException();
  return;
}

void Server::SwitchToSafeMode() {
  MS_LOG(INFO) << "Server switch to safemode.";
  safemode_ = true;
}

void Server::CancelSafeMode() {
  MS_LOG(INFO) << "Server cancel safemode.";
  safemode_ = false;
}

bool Server::IsSafeMode() const { return safemode_.load(); }

void Server::WaitExitSafeMode() const {
  while (safemode_.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kThreadSleepTime));
  }
}

void Server::InitServerContext() {
  ps::PSContext::instance()->GenerateResetterRound();
  scheduler_ip_ = ps::PSContext::instance()->scheduler_host();
  scheduler_port_ = ps::PSContext::instance()->scheduler_port();
  worker_num_ = ps::PSContext::instance()->initial_worker_num();
  server_num_ = ps::PSContext::instance()->initial_server_num();
  std::string encrypt_type = ps::PSContext::instance()->encrypt_type();
  if (encrypt_type == ps::kPWEncryptType && server_num_ > 1) {
    MS_LOG(EXCEPTION) << "Only single server is supported for PW_ENCRYPT now, but got server_num is:." << server_num_;
    return;
  }
  return;
}

void Server::InitCluster() {
  server_node_ = std::make_shared<ps::core::ServerNode>();
  MS_EXCEPTION_IF_NULL(server_node_);
  task_executor_ = std::make_shared<ps::core::TaskExecutor>(kExecutorThreadPoolSize);
  MS_EXCEPTION_IF_NULL(task_executor_);
  if (!InitCommunicatorWithServer()) {
    MS_LOG(EXCEPTION) << "Initializing cross-server communicator failed.";
    return;
  }
  if (!InitCommunicatorWithWorker()) {
    MS_LOG(EXCEPTION) << "Initializing worker-server communicator failed.";
    return;
  }
  return;
}

bool Server::InitCommunicatorWithServer() {
  MS_EXCEPTION_IF_NULL(task_executor_);
  MS_EXCEPTION_IF_NULL(server_node_);
  communicator_with_server_ = server_node_->GetOrCreateTcpComm(scheduler_ip_, static_cast<int16_t>(scheduler_port_),
                                                               worker_num_, server_num_, task_executor_);
  MS_EXCEPTION_IF_NULL(communicator_with_server_);
  g_communicator_with_server = communicator_with_server_;
  return true;
}

bool Server::InitCommunicatorWithWorker() {
  MS_EXCEPTION_IF_NULL(server_node_);
  MS_EXCEPTION_IF_NULL(task_executor_);
  if (!use_tcp_ && !use_http_) {
    MS_LOG(EXCEPTION) << "At least one type of protocol should be set.";
    return false;
  }
  if (use_tcp_) {
    MS_EXCEPTION_IF_NULL(communicator_with_server_);
    auto tcp_comm = communicator_with_server_;
    MS_EXCEPTION_IF_NULL(tcp_comm);
    communicators_with_worker_.push_back(tcp_comm);
  }
  if (use_http_) {
    auto http_comm = server_node_->GetOrCreateHttpComm(server_node_->BoundIp(), http_port_, task_executor_);
    MS_EXCEPTION_IF_NULL(http_comm);
    communicators_with_worker_.push_back(http_comm);
  }
  g_communicators_with_worker = communicators_with_worker_;
  return true;
}

void Server::InitIteration() {
  iteration_ = &Iteration::GetInstance();
  MS_EXCEPTION_IF_NULL(iteration_);

  // 1.Add rounds to the iteration according to the server mode.
  for (const RoundConfig &config : rounds_config_) {
    std::shared_ptr<Round> round =
      std::make_shared<Round>(config.name, config.check_timeout, config.time_window, config.check_count,
                              config.threshold_count, config.server_num_as_threshold);
    MS_LOG(INFO) << "Add round " << config.name << ", check_timeout: " << config.check_timeout
                 << ", time window: " << config.time_window << ", check_count: " << config.check_count
                 << ", threshold: " << config.threshold_count
                 << ", server_num_as_threshold: " << config.server_num_as_threshold;
    iteration_->AddRound(round);
  }

#ifdef ENABLE_ARMOUR
  std::string encrypt_type = ps::PSContext::instance()->encrypt_type();
  if (encrypt_type == ps::kPWEncryptType) {
    cipher_exchange_keys_cnt_ = cipher_config_.exchange_keys_threshold;
    cipher_get_keys_cnt_ = cipher_config_.get_keys_threshold;
    cipher_share_secrets_cnt_ = cipher_config_.share_secrets_threshold;
    cipher_get_secrets_cnt_ = cipher_config_.get_secrets_threshold;
    cipher_get_clientlist_cnt_ = cipher_config_.client_list_threshold;
    cipher_reconstruct_secrets_up_cnt_ = cipher_config_.reconstruct_secrets_threshold;
    cipher_reconstruct_secrets_down_cnt_ = cipher_config_.reconstruct_secrets_threshold - 1;
    cipher_time_window_ = cipher_config_.cipher_time_window;

    MS_LOG(INFO) << "Initializing cipher:";
    MS_LOG(INFO) << " cipher_exchange_keys_cnt_: " << cipher_exchange_keys_cnt_
                 << " cipher_get_keys_cnt_: " << cipher_get_keys_cnt_
                 << " cipher_share_secrets_cnt_: " << cipher_share_secrets_cnt_;
    MS_LOG(INFO) << " cipher_get_secrets_cnt_: " << cipher_get_secrets_cnt_
                 << " cipher_get_clientlist_cnt_: " << cipher_get_clientlist_cnt_
                 << " cipher_reconstruct_secrets_up_cnt_: " << cipher_reconstruct_secrets_up_cnt_
                 << " cipher_reconstruct_secrets_down_cnt_: " << cipher_reconstruct_secrets_down_cnt_
                 << " cipher_time_window_: " << cipher_time_window_;
  }
#endif

  // 2.Initialize all the rounds.
  TimeOutCb time_out_cb = std::bind(&Iteration::NotifyNext, iteration_, std::placeholders::_1, std::placeholders::_2);
  FinishIterCb finish_iter_cb =
    std::bind(&Iteration::NotifyNext, iteration_, std::placeholders::_1, std::placeholders::_2);
  iteration_->InitRounds(communicators_with_worker_, time_out_cb, finish_iter_cb);
  return;
}

void Server::InitCipher() {
#ifdef ENABLE_ARMOUR
  cipher_init_ = &armour::CipherInit::GetInstance();
  int cipher_t = SizeToInt(cipher_reconstruct_secrets_down_cnt_);
  unsigned char cipher_p[SECRET_MAX_LEN] = {0};
  const int cipher_g = 1;
  float dp_eps = ps::PSContext::instance()->dp_eps();
  float dp_delta = ps::PSContext::instance()->dp_delta();
  float dp_norm_clip = ps::PSContext::instance()->dp_norm_clip();
  std::string encrypt_type = ps::PSContext::instance()->encrypt_type();

  mindspore::armour::CipherPublicPara param;
  param.g = cipher_g;
  param.t = cipher_t;
  int ret = memcpy_s(param.p, SECRET_MAX_LEN, cipher_p, sizeof(cipher_p));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    return;
  }
  param.dp_delta = dp_delta;
  param.dp_eps = dp_eps;
  param.dp_norm_clip = dp_norm_clip;
  param.encrypt_type = encrypt_type;

  BIGNUM *prim = BN_new();
  if (prim == NULL) {
    MS_LOG(EXCEPTION) << "new bn failed.";
    ret = -1;
  } else {
    ret = mindspore::armour::GetPrime(prim);
  }
  if (ret == 0) {
    (void)BN_bn2bin(prim, reinterpret_cast<uint8_t *>(param.prime));
  } else {
    MS_LOG(EXCEPTION) << "Get prime failed.";
  }
  if (prim != NULL) {
    BN_clear_free(prim);
  }

  (void)cipher_init_->Init(param, 0, cipher_exchange_keys_cnt_, cipher_get_keys_cnt_, cipher_share_secrets_cnt_,
                           cipher_get_secrets_cnt_, cipher_get_clientlist_cnt_, cipher_reconstruct_secrets_up_cnt_);
#endif
}

void Server::RegisterCommCallbacks() {
  // The message callbacks of round kernels are already set in method InitIteration, so here we don't need to register
  // rounds' callbacks.
  MS_EXCEPTION_IF_NULL(server_node_);
  MS_EXCEPTION_IF_NULL(iteration_);

  auto tcp_comm = std::dynamic_pointer_cast<ps::core::TcpCommunicator>(communicator_with_server_);
  MS_EXCEPTION_IF_NULL(tcp_comm);

  // Set message callbacks for server-to-server communication.
  DistributedMetadataStore::GetInstance().RegisterMessageCallback(tcp_comm);
  DistributedCountService::GetInstance().RegisterMessageCallback(tcp_comm);
  iteration_->RegisterMessageCallback(tcp_comm);
  iteration_->RegisterEventCallback(server_node_);

  // Set exception event callbacks for server.
  RegisterExceptionEventCallback(tcp_comm);
  // Set message callbacks for server.
  RegisterMessageCallback(tcp_comm);

  if (!server_node_->InitFollowerScaler()) {
    MS_LOG(EXCEPTION) << "Initializing follower elastic scaler failed.";
    return;
  }
  // Set scaling barriers before scaling.
  server_node_->RegisterFollowerScalerBarrierBeforeScaleOut("ServerPipeline",
                                                            std::bind(&Server::ProcessBeforeScalingOut, this));
  server_node_->RegisterFollowerScalerBarrierBeforeScaleIn("ServerPipeline",
                                                           std::bind(&Server::ProcessBeforeScalingIn, this));
  // Set handlers after scheduler scaling operations are done.
  server_node_->RegisterFollowerScalerHandlerAfterScaleOut("ServerPipeline",
                                                           std::bind(&Server::ProcessAfterScalingOut, this));
  server_node_->RegisterFollowerScalerHandlerAfterScaleIn("ServerPipeline",
                                                          std::bind(&Server::ProcessAfterScalingIn, this));
}

void Server::RegisterExceptionEventCallback(const std::shared_ptr<ps::core::TcpCommunicator> &communicator) {
  MS_EXCEPTION_IF_NULL(communicator);
  communicator->RegisterEventCallback(ps::core::ClusterEvent::SCHEDULER_TIMEOUT, [&]() {
    MS_LOG(ERROR) << "Event SCHEDULER_TIMEOUT is captured. This is because scheduler node is finalized or crashed.";
    safemode_ = true;
    (void)std::for_each(communicators_with_worker_.begin(), communicators_with_worker_.end(),
                        [](const std::shared_ptr<ps::core::CommunicatorBase> &communicator) {
                          MS_ERROR_IF_NULL_WO_RET_VAL(communicator);
                          (void)communicator->Stop();
                        });

    MS_ERROR_IF_NULL_WO_RET_VAL(communicator_with_server_);
    (void)communicator_with_server_->Stop();
  });

  communicator->RegisterEventCallback(ps::core::ClusterEvent::NODE_TIMEOUT, [&]() {
    MS_LOG(ERROR)
      << "Event NODE_TIMEOUT is captured. This is because some server nodes are finalized or crashed after the "
         "network building phase.";
    safemode_ = true;
    (void)std::for_each(communicators_with_worker_.begin(), communicators_with_worker_.end(),
                        [](const std::shared_ptr<ps::core::CommunicatorBase> &communicator) {
                          MS_ERROR_IF_NULL_WO_RET_VAL(communicator);
                          (void)communicator->Stop();
                        });

    MS_ERROR_IF_NULL_WO_RET_VAL(communicator_with_server_);
    (void)communicator_with_server_->Stop();
  });
}

void Server::RegisterMessageCallback(const std::shared_ptr<ps::core::TcpCommunicator> &communicator) {
  MS_EXCEPTION_IF_NULL(communicator);
  // Register handler for restful requests receviced by scheduler.
  communicator->RegisterMsgCallBack("enableFLS",
                                    std::bind(&Server::HandleEnableServerRequest, this, std::placeholders::_1));
  communicator->RegisterMsgCallBack("disableFLS",
                                    std::bind(&Server::HandleDisableServerRequest, this, std::placeholders::_1));
  communicator->RegisterMsgCallBack("newInstance",
                                    std::bind(&Server::HandleNewInstanceRequest, this, std::placeholders::_1));
  communicator->RegisterMsgCallBack("queryInstance",
                                    std::bind(&Server::HandleQueryInstanceRequest, this, std::placeholders::_1));
}

void Server::InitExecutor() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  if (executor_threshold_ == 0) {
    MS_LOG(EXCEPTION) << "The executor's threshold should greater than 0.";
    return;
  }
  // The train engine instance is used in both push-type and pull-type kernels,
  // so the required_cnt of these kernels must be the same as executor_threshold_.
  MS_LOG(INFO) << "Required count for push-type and pull-type kernels is " << executor_threshold_;
  Executor::GetInstance().Initialize(func_graph_, executor_threshold_);
  ModelStore::GetInstance().Initialize();
  return;
}

void Server::RegisterRoundKernel() {
  MS_EXCEPTION_IF_NULL(iteration_);
  auto &rounds = iteration_->rounds();
  if (rounds.empty()) {
    MS_LOG(EXCEPTION) << "Server has no round registered.";
    return;
  }

  for (auto &round : rounds) {
    MS_EXCEPTION_IF_NULL(round);
    const std::string &name = round->name();
    std::shared_ptr<kernel::RoundKernel> round_kernel = kernel::RoundKernelFactory::GetInstance().Create(name);
    if (round_kernel == nullptr) {
      MS_LOG(EXCEPTION) << "Round kernel for round " << name << " is not registered.";
      return;
    }

    // For some round kernels, the threshold count should be set.
    round_kernel->InitKernel(round->threshold_count());
    round->BindRoundKernel(round_kernel);
  }
  return;
}

void Server::InitMetrics() {
  if (server_node_->rank_id() == kLeaderServerRank) {
    MS_EXCEPTION_IF_NULL(iteration_);
    std::shared_ptr<IterationMetrics> iteration_metrics =
      std::make_shared<IterationMetrics>(ps::PSContext::instance()->config_file_path());
    if (!iteration_metrics->Initialize()) {
      MS_LOG(WARNING) << "Initializing metrics failed.";
      return;
    }
    iteration_->set_metrics(iteration_metrics);
  }
}

void Server::StartCommunicator() {
  if (communicators_with_worker_.empty()) {
    MS_LOG(EXCEPTION) << "Communicators for communication with worker is empty.";
    return;
  }

  MS_EXCEPTION_IF_NULL(server_node_);
  MS_EXCEPTION_IF_NULL(communicator_with_server_);
  MS_LOG(INFO) << "Start communicator with server.";
  if (!communicator_with_server_->Start()) {
    MS_LOG(EXCEPTION) << "Starting communicator with server failed.";
    return;
  }
  DistributedMetadataStore::GetInstance().Initialize(server_node_);
  CollectiveOpsImpl::GetInstance().Initialize(server_node_);
  DistributedCountService::GetInstance().Initialize(server_node_, kLeaderServerRank);
  MS_LOG(INFO) << "This server rank is " << server_node_->rank_id();

  MS_LOG(INFO) << "Start communicator with worker.";
  (void)std::for_each(communicators_with_worker_.begin(), communicators_with_worker_.end(),
                      [](const std::shared_ptr<ps::core::CommunicatorBase> &communicator) {
                        MS_ERROR_IF_NULL_WO_RET_VAL(communicator);
                        if (!communicator->Start()) {
                          MS_LOG(EXCEPTION) << "Starting communicator with worker failed.";
                        }
                      });
}

void Server::ProcessBeforeScalingOut() {
  MS_ERROR_IF_NULL_WO_RET_VAL(iteration_);
  iteration_->ScalingBarrier();
  safemode_ = true;
}

void Server::ProcessBeforeScalingIn() {
  MS_ERROR_IF_NULL_WO_RET_VAL(iteration_);
  iteration_->ScalingBarrier();
  safemode_ = true;
}

void Server::ProcessAfterScalingOut() {
  std::unique_lock<std::mutex> lock(scaling_mtx_);
  MS_ERROR_IF_NULL_WO_RET_VAL(server_node_);
  if (!DistributedMetadataStore::GetInstance().ReInitForScaling()) {
    MS_LOG(WARNING) << "DistributedMetadataStore reinitializing failed.";
  }
  if (!CollectiveOpsImpl::GetInstance().ReInitForScaling()) {
    MS_LOG(WARNING) << "DistributedMetadataStore reinitializing failed.";
  }
  if (!DistributedCountService::GetInstance().ReInitForScaling()) {
    MS_LOG(WARNING) << "DistributedCountService reinitializing failed.";
  }
  if (!iteration_->ReInitForScaling(IntToUint(server_node_->server_num()), server_node_->rank_id())) {
    MS_LOG(WARNING) << "Iteration reinitializing failed.";
  }
  if (!Executor::GetInstance().ReInitForScaling()) {
    MS_LOG(WARNING) << "Executor reinitializing failed.";
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(kServerSleepTimeForNetworking));
  safemode_ = false;
}

void Server::ProcessAfterScalingIn() {
  std::unique_lock<std::mutex> lock(scaling_mtx_);
  MS_ERROR_IF_NULL_WO_RET_VAL(server_node_);
  if (server_node_->rank_id() == UINT32_MAX) {
    MS_LOG(WARNING) << "This server the one to be scaled in. Server need to wait SIGTERM to exit.";
    return;
  }

  // If the server is not the one to be scaled in, reintialize modules and recover service.
  if (!DistributedMetadataStore::GetInstance().ReInitForScaling()) {
    MS_LOG(WARNING) << "DistributedMetadataStore reinitializing failed.";
  }
  if (!CollectiveOpsImpl::GetInstance().ReInitForScaling()) {
    MS_LOG(WARNING) << "DistributedMetadataStore reinitializing failed.";
  }
  if (!DistributedCountService::GetInstance().ReInitForScaling()) {
    MS_LOG(WARNING) << "DistributedCountService reinitializing failed.";
  }
  if (!iteration_->ReInitForScaling(IntToUint(server_node_->server_num()), server_node_->rank_id())) {
    MS_LOG(WARNING) << "Iteration reinitializing failed.";
  }
  if (!Executor::GetInstance().ReInitForScaling()) {
    MS_LOG(WARNING) << "Executor reinitializing failed.";
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(kServerSleepTimeForNetworking));
  safemode_ = false;
}

void Server::HandleEnableServerRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  MS_ERROR_IF_NULL_WO_RET_VAL(iteration_);
  MS_ERROR_IF_NULL_WO_RET_VAL(communicator_with_server_);
  auto tcp_comm = std::dynamic_pointer_cast<ps::core::TcpCommunicator>(communicator_with_server_);
  MS_ERROR_IF_NULL_WO_RET_VAL(tcp_comm);

  std::string result_message = "";
  bool result = iteration_->EnableServerInstance(&result_message);
  nlohmann::json response;
  response["result"] = result;
  response["message"] = result_message;
  if (!tcp_comm->SendResponse(response.dump().c_str(), response.dump().size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }
}

void Server::HandleDisableServerRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  MS_ERROR_IF_NULL_WO_RET_VAL(iteration_);
  MS_ERROR_IF_NULL_WO_RET_VAL(communicator_with_server_);
  auto tcp_comm = std::dynamic_pointer_cast<ps::core::TcpCommunicator>(communicator_with_server_);
  MS_ERROR_IF_NULL_WO_RET_VAL(tcp_comm);

  std::string result_message = "";
  bool result = iteration_->DisableServerInstance(&result_message);
  nlohmann::json response;
  response["result"] = result;
  response["message"] = result_message;
  if (!tcp_comm->SendResponse(response.dump().c_str(), response.dump().size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }
}

void Server::HandleNewInstanceRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  MS_ERROR_IF_NULL_WO_RET_VAL(iteration_);
  MS_ERROR_IF_NULL_WO_RET_VAL(communicator_with_server_);
  auto tcp_comm = std::dynamic_pointer_cast<ps::core::TcpCommunicator>(communicator_with_server_);
  MS_ERROR_IF_NULL_WO_RET_VAL(tcp_comm);

  MS_ERROR_IF_NULL_WO_RET_VAL(message->data());
  std::string hyper_params_str(static_cast<const char *>(message->data()), message->len());
  nlohmann::json new_instance_json;
  nlohmann::json response;
  try {
    new_instance_json = nlohmann::json::parse(hyper_params_str);
  } catch (const std::exception &e) {
    response["result"] = false;
    response["message"] = "The hyper-parameter data is not in json format.";
    if (!tcp_comm->SendResponse(response.dump().c_str(), response.dump().size(), message)) {
      MS_LOG(ERROR) << "Sending response failed.";
      return;
    }
  }

  std::string result_message = "";
  bool result = iteration_->NewInstance(new_instance_json, &result_message);
  response["result"] = result;
  response["message"] = result_message;
  if (!tcp_comm->SendResponse(response.dump().c_str(), response.dump().size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }
}

void Server::HandleQueryInstanceRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  nlohmann::basic_json<std::map, std::vector, std::string, bool, int64_t, uint64_t, float> response;
  response["start_fl_job_threshold"] = ps::PSContext::instance()->start_fl_job_threshold();
  response["start_fl_job_time_window"] = ps::PSContext::instance()->start_fl_job_time_window();
  response["update_model_ratio"] = ps::PSContext::instance()->update_model_ratio();
  response["update_model_time_window"] = ps::PSContext::instance()->update_model_time_window();
  response["fl_iteration_num"] = ps::PSContext::instance()->fl_iteration_num();
  response["client_epoch_num"] = ps::PSContext::instance()->client_epoch_num();
  response["client_batch_size"] = ps::PSContext::instance()->client_batch_size();
  response["client_learning_rate"] = ps::PSContext::instance()->client_learning_rate();
  auto tcp_comm = std::dynamic_pointer_cast<ps::core::TcpCommunicator>(communicator_with_server_);
  MS_ERROR_IF_NULL_WO_RET_VAL(tcp_comm);
  if (!tcp_comm->SendResponse(response.dump().c_str(), response.dump().size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
