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

#ifndef MINDSPORE_CCSRC_PS_CONSTANTS_H_
#define MINDSPORE_CCSRC_PS_CONSTANTS_H_

#include <limits.h>

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <functional>
#include "utils/shape_utils.h"

namespace mindspore {
namespace distributed::persistent {
template <typename T>
class Data;
template <typename T>
class PersistentData;
}  // namespace distributed::persistent
namespace ps {
constexpr char kEnvCommType[] = "MS_COMM_TYPE";
constexpr char kEnvInterface[] = "MS_INTERFACE";
constexpr char kEnvPServerNum[] = "MS_SERVER_NUM";
constexpr char kEnvWorkerNum[] = "MS_WORKER_NUM";
constexpr char kEnvSchedulerHost[] = "MS_SCHED_HOST";
constexpr char kEnvSchedulerPort[] = "MS_SCHED_PORT";
constexpr char kEnvSchedulerManagePort[] = "MS_SCHED_MANAGE_PORT";
constexpr char kEnvNodeId[] = "MS_NODE_ID";

constexpr char kCommTypeOfIBVerbs[] = "ibverbs";
constexpr char kRoleOfPServer[] = "server";
constexpr char kRoleOfWorker[] = "worker";
constexpr char kRoleOfScheduler[] = "scheduler";

constexpr char kLearningRate[] = "learning_rate";
constexpr char kMomentum[] = "momentum";

constexpr char kApplyMomentum[] = "ApplyMomentum";
constexpr char kSparseAdam[] = "Adam";
constexpr char kSparseLazyAdam[] = "LazyAdam";
constexpr char kSparseFtrl[] = "Ftrl";
constexpr char kApplyMomentumOp[] = "Momentum";
constexpr char kSparseAdamOp[] = "Adam";
constexpr char kSparseLazyAdamOp[] = "LazyAdam";
constexpr char kSparseFtrlOp[] = "FTRL";

constexpr char kCertificateChain[] = "server.crt";
constexpr char kPrivateKey[] = "server.key";
constexpr char kCAcrt[] = "ca.crt";

constexpr char kKeys[] = "keys";
constexpr char kShapes[] = "shapes";
constexpr char kParamNames[] = "param_names";
constexpr char kRecoverFunc[] = "recover_function";
constexpr char kRecoverEmbedding[] = "RecoverEmbedding";
constexpr char kCurrentDirOfServer[] = "./server_";
constexpr char kParamWithKey[] = "_parameter_key_";

constexpr int64_t kInitWeightsCmd = 10;
constexpr int64_t kInitWeightToOptimIdCmd = 11;
constexpr int64_t kInitOptimInputsShapeCmd = 12;
constexpr int64_t kInitKeyToPushNodeIdCmd = 13;
constexpr int64_t kInitEmbeddingsCmd = 20;
constexpr int64_t kUpdateEmbeddingsCmd = 21;
constexpr int64_t kCheckReadyForPushCmd = 25;
constexpr int64_t kCheckReadyForPullCmd = 26;
constexpr int64_t kEmbeddingLookupCmd = 30;
constexpr int64_t kFinalizeCmd = 40;
constexpr int64_t kPushCmd = 50;
constexpr int64_t kPullCmd = 51;

constexpr size_t kInvalidKey = UINT64_MAX;
constexpr int64_t kInvalidID = -1;

constexpr uint32_t kMaxMessageSize = static_cast<uint32_t>(100 * (uint32_t(1) << 20));
constexpr char kServerNum[] = "server_num";
constexpr char kWorkerNum[] = "worker_num";
constexpr char kNodesIds[] = "node_ids";
constexpr char kNodeId[] = "node_id";

constexpr char kSuccessCode[] = "0";
constexpr char kErrorCode[] = "1";

constexpr int64_t kSubmitTaskIntervalInMs = 1;
constexpr int64_t kMaxTaskNum = 10240;
constexpr int64_t kSubmitTimeOutInMs = 30000;
constexpr int64_t kRetryCount = 60;
constexpr int64_t kRetryIntervalInMs = 10;

constexpr int64_t kThreadNum = 32;
constexpr int64_t kGradIndex = 0;
constexpr int64_t kIndiceIndex = 1;
constexpr int64_t kFirstDimSize = 2;
constexpr int64_t kOutDimSize = 3;

constexpr int64_t kBase = 10;
constexpr float kStdDev = 0.01;
// The timeout period for the scale in node to send the finish message to scheduler.
constexpr uint32_t kScaleInTimeoutInSenconds = 30;
// The number of retries to determine whether all nodes are successfully registered.
constexpr uint32_t kCheckRegisteredRetryCount = 30;
// The timeout interval for judging whether all nodes are successfully registered.
constexpr uint32_t kCheckRegisteredIntervalInMs = 1000;

constexpr int64_t kSparseLazyAdamIndex = 2;
constexpr int64_t kSparseFtrlIndex = 3;
constexpr int64_t kSparseGradIndex = 6;
constexpr int64_t kSparseIndiceIndex = 7;

constexpr int64_t kHeartbeatTimes = 2;
constexpr int64_t kGradValue = -100;
// Whether to support recovery.
constexpr char kIsRecovery[] = "is_recovery";
// The type of persistent storage, currently only supports file storage.
constexpr char kStoreType[] = "storage_type";
// The file used to storage metadata.
constexpr char kStoreFilePath[] = "storage_file_path";
// The file used to storage scheduler metadata.
constexpr char kSchedulerStoreFilePath[] = "scheduler_storage_file_path";
// 1 indicates that the persistent storage type is file.
constexpr char kFileStorage[] = "1";
// The recovery key of json_config.
constexpr char kKeyRecovery[] = "recovery";
constexpr char kRecoveryWorkerNum[] = "worker_num";
constexpr char kRecoveryServerNum[] = "server_num";
constexpr char kRecoverySchedulerIp[] = "scheduler_ip";
constexpr char kRecoverySchedulerPort[] = "scheduler_port";
constexpr char kRecoveryTotalNodeNum[] = "total_node_num";
constexpr char kRecoveryNextWorkerRankId[] = "next_worker_rank_id";
constexpr char kRecoveryNextServerRankId[] = "next_server_rank_id";
constexpr char kRecoveryRegisteredNodesInfos[] = "node_ids";
constexpr char kRecoveryClusterState[] = "cluster_state";

constexpr char kServerCertPath[] = "server_cert_path";
constexpr char kServerPassword[] = "server_password";
constexpr char kCrlPath[] = "crl_path";
constexpr char kClientCertPath[] = "client_cert_path";
constexpr char kClientPassword[] = "client_password";
constexpr char kCaCertPath[] = "ca_cert_path";
constexpr char kCipherList[] = "cipher_list";
constexpr char kCertCheckInterval[] = "cert_check_interval_in_hour";
// 7 * 24
constexpr int64_t kCertCheckIntervalInHour = 168;
constexpr char kCertExpireWarningTime[] = "cert_expire_warning_time_in_day";
// 90
constexpr int64_t kCertExpireWarningTimeInDay = 90;
constexpr char kConnectionNum[] = "connection_num";
constexpr int64_t kConnectionNumDefault = 10000;
constexpr char kLocalIp[] = "127.0.0.1";

constexpr int64_t kJanuary = 1;
constexpr int64_t kSeventyYear = 70;
constexpr int64_t kHundredYear = 100;
constexpr int64_t kThousandYear = 1000;
constexpr int64_t kBaseYear = 1900;
constexpr int64_t kMinWarningTime = 7;
constexpr int64_t kMaxWarningTime = 180;

constexpr int64_t kLength = 100;
constexpr int64_t kMaxPort = 65535;
constexpr int64_t kSecurityLevel = 3;

constexpr char kTcpCommunicator[] = "TCP";
constexpr char kHttpCommunicator[] = "HTTP";

constexpr char kServerCert[] = "server.p12";
constexpr char kClientCert[] = "client.p12";
constexpr char kCaCert[] = "ca.crt";
constexpr char kColon = ':';
const std::map<std::string, size_t> kCiphers = {
  {"ECDHE-RSA-AES128-GCM-SHA256", 0},   {"ECDHE-ECDSA-AES128-GCM-SHA256", 1}, {"ECDHE-RSA-AES256-GCM-SHA384", 2},
  {"ECDHE-ECDSA-AES256-GCM-SHA384", 3}, {"ECDHE-RSA-CHACHA20-POLY1305", 4},   {"ECDHE-PSK-CHACHA20-POLY1305", 5},
  {"ECDHE-ECDSA-AES128-CCM", 6},        {"ECDHE-ECDSA-AES256-CCM", 7},        {"ECDHE-ECDSA-CHACHA20-POLY1305", 8}};

using DataPtr = std::unique_ptr<uint8_t[]>;
using VectorPtr = std::shared_ptr<std::vector<unsigned char>>;
using Key = size_t;
using Keys = std::vector<Key>;
using Values = std::vector<float>;
using ValuesPtr = std::shared_ptr<Values>;
using Weight = distributed::persistent::Data<float>;
using PersistentWeight = distributed::persistent::PersistentData<float>;
using Grad = std::vector<float>;
using LookupIds = std::vector<Key>;
using Lengths = std::vector<int>;
using WeightPtr = std::shared_ptr<Weight>;
using PersistentWeightPtr = std::shared_ptr<PersistentWeight>;
using GradPtr = std::shared_ptr<Grad>;
using InputsShape = std::vector<std::shared_ptr<ShapeVector>>;
using InputsShapePtr = std::shared_ptr<std::vector<std::shared_ptr<ShapeVector>>>;

constexpr size_t INDEX_NOT_SEND = UINT_MAX;
using OptimOriginIdx = std::map<std::string, size_t>;
using OptimPSSendIdx = std::map<std::string, size_t>;

using EventCallback = std::function<void(void)>;

const OptimOriginIdx kMomentumOriginIdx = {{"weight", 0}, {"accum", 1}, {"lr", 2}, {"grad", 3}, {"momentum", 4}};
const OptimPSSendIdx kMomentumPSSendIdx = {
  {"weight", INDEX_NOT_SEND}, {"accum", INDEX_NOT_SEND}, {"lr", 0}, {"grad", 1}, {"momentum", 2}};

const OptimOriginIdx kSparseAdamOriginIdx = {{"weight", 0},      {"m", 1},    {"v", 2},       {"beta1_power", 3},
                                             {"beta2_power", 4}, {"lr", 5},   {"beta1", 6},   {"beta2", 7},
                                             {"eps", 8},         {"grad", 9}, {"indices", 10}};
const OptimPSSendIdx kSparseAdamPSSendIdx = {{"weight", INDEX_NOT_SEND},
                                             {"m", INDEX_NOT_SEND},
                                             {"v", INDEX_NOT_SEND},
                                             {"beta1_power", 0},
                                             {"beta2_power", 1},
                                             {"lr", 2},
                                             {"beta1", 3},
                                             {"beta2", 4},
                                             {"eps", 5},
                                             {"grad", 6},
                                             {"indices", 7}};

const OptimOriginIdx kSparseFtrlOriginIdx = {{"weight", 0}, {"accum", 1}, {"linear", 2}, {"grad", 3}, {"indices", 4}};
const OptimPSSendIdx kSparseFtrlPSSendIdx = {
  {"weight", INDEX_NOT_SEND}, {"accum", INDEX_NOT_SEND}, {"linear", INDEX_NOT_SEND}, {"grad", 0}, {"indices", 1}};

const std::map<std::string, OptimOriginIdx> kOptimToOriginIdx = {{kApplyMomentum, kMomentumOriginIdx},
                                                                 {kSparseAdam, kSparseAdamOriginIdx},
                                                                 {kSparseLazyAdam, kSparseAdamOriginIdx},
                                                                 {kSparseFtrl, kSparseFtrlOriginIdx}};
const std::map<std::string, OptimOriginIdx> kOptimToPSSendIdx = {{kApplyMomentum, kMomentumPSSendIdx},
                                                                 {kSparseAdam, kSparseAdamPSSendIdx},
                                                                 {kSparseLazyAdam, kSparseAdamPSSendIdx},
                                                                 {kSparseFtrl, kSparseFtrlPSSendIdx}};

// The barrier function which should be called before doing scaling out/in operations.
// It's easy for us to scale out/in nodes after one iteration is completed and keep consistent.
using BarrierBeforeScaleOut = std::function<void(void)>;
using BarrierBeforeScaleIn = std::function<void(void)>;

// These handlers helps worker/server node to reinitialize or recover data after scaling out/in operation of scheduler
// is done.
using HandlerAfterScaleOut = std::function<void(void)>;
using HandlerAfterScaleIn = std::function<void(void)>;
using HandlerAfterScaleOutRollback = std::function<void(void)>;

constexpr char kClusterNotReady[] =
  "The Scheduler's connections are not equal with total node num, Maybe this is because some server nodes are drop "
  "out or scale in nodes has not been recycled.";
constexpr char kJobNotReady[] = "The server's training job is not ready.";
constexpr char kClusterSafeMode[] = "The cluster is in safemode.";
constexpr char kJobNotAvailable[] = "The server's training job is disabled or finished.";

enum class UserDefineEvent { kIterationRunning = 0, kIterationCompleted, kNodeTimeout };

#define EXC_IF_VEC_IDX_OOB(vec, idx)                                                            \
  do {                                                                                          \
    size_t vec_size = vec.size();                                                               \
    if (idx >= vec_size) {                                                                      \
      MS_LOG(EXCEPTION) << "Vector " << #vec << " size is " << vec_size << ". So index " << idx \
                        << " is out of bound.";                                                 \
    }                                                                                           \
  } while (0)

#define ERROR_STATUS(result, code, message)       \
  do {                                            \
    MS_LOG(ERROR) << message;                     \
    result = RequestProcessResult(code, message); \
  } while (0)

#define CHECK_RETURN_TYPE(_condition)                    \
  do {                                                   \
    if (!(_condition)) {                                 \
      MS_LOG(ERROR) << "Parse protobuf message failed."; \
    }                                                    \
  } while (false)
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CONSTANTS_H_
