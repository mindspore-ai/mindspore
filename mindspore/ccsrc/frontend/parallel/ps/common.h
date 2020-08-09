/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_COMMON_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_COMMON_H_

#include <iostream>
#include <vector>
#include <memory>
#include "ps/ps.h"

namespace mindspore {
namespace parallel {
namespace ps {
constexpr char kEnvCommType[] = "MS_COMM_TYPE";
constexpr char kEnvInterface[] = "MS_INTERFACE";
constexpr char kEnvPServerNum[] = "MS_SERVER_NUM";
constexpr char kEnvWorkerNum[] = "MS_WORKER_NUM";
constexpr char kEnvSchedulerHost[] = "MS_SCHED_HOST";
constexpr char kEnvSchedulerPort[] = "MS_SCHED_PORT";

constexpr char kEnvRole[] = "MS_ROLE";
constexpr char kEnvRoleOfPServer[] = "MS_PSERVER";
constexpr char kEnvRoleOfWorker[] = "MS_WORKER";
constexpr char kEnvRoleOfScheduler[] = "MS_SCHED";

constexpr char kDmlcCommType[] = "DMLC_PS_VAN_TYPE";
constexpr char kDmlcInterface[] = "DMLC_INTERFACE";
constexpr char kDmlcPServerNum[] = "DMLC_NUM_SERVER";
constexpr char kDmlcWorkerNum[] = "DMLC_NUM_WORKER";
constexpr char kDmlcRole[] = "DMLC_ROLE";
constexpr char kDmlcSchedulerHost[] = "DMLC_PS_ROOT_URI";
constexpr char kDmlcSchedulerPort[] = "DMLC_PS_ROOT_PORT";

constexpr char kCommTypeOfIBVerbs[] = "ibverbs";
constexpr char kCommTypeOfTCP[] = "zmq";
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

constexpr int kInitWeightsCmd = 10;
constexpr int kInitWeightToOptimIdCmd = 11;
constexpr int kInitOptimInputsShapeCmd = 12;
constexpr int kInitKeyToPushNodeIdCmd = 13;
constexpr int kInitEmbeddingsCmd = 20;
constexpr int kCheckReadyForPushCmd = 25;
constexpr int kCheckReadyForPullCmd = 26;
constexpr int kEmbeddingLookupCmd = 30;
constexpr int kFinalizeCmd = 40;

constexpr size_t kInvalidKey = UINT64_MAX;
constexpr int kInvalidID = -1;

using Key = ::ps::Key;
using Keys = ::ps::SArray<Key>;
using Values = ::ps::SArray<float>;
using ValuesPtr = std::shared_ptr<Values>;
using Weight = ::ps::SArray<float>;
using Grad = ::ps::SArray<float>;
using LookupIds = ::ps::SArray<Key>;
using Lengths = ::ps::SArray<int>;
using WeightPtr = std::shared_ptr<Weight>;
using GradPtr = std::shared_ptr<Grad>;
using InputsShape = std::vector<std::shared_ptr<std::vector<size_t>>>;
using InputsShapePtr = std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>>;
}  // namespace ps
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_COMMON_H_
