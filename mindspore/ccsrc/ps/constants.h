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

namespace mindspore {
namespace ps {
constexpr char kEnvCommType[] = "MS_COMM_TYPE";
constexpr char kEnvInterface[] = "MS_INTERFACE";
constexpr char kEnvPServerNum[] = "MS_SERVER_NUM";
constexpr char kEnvWorkerNum[] = "MS_WORKER_NUM";
constexpr char kEnvSchedulerHost[] = "MS_SCHED_HOST";
constexpr char kEnvSchedulerPort[] = "MS_SCHED_PORT";

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

using DataPtr = std::shared_ptr<unsigned char[]>;
using VectorPtr = std::shared_ptr<std::vector<unsigned char>>;
using Key = uint64_t;
using Keys = std::vector<Key>;
using Values = std::vector<float>;
using ValuesPtr = std::shared_ptr<Values>;
using Weight = std::vector<float>;
using Grad = std::vector<float>;
using LookupIds = std::vector<Key>;
using Lengths = std::vector<int>;
using WeightPtr = std::shared_ptr<Weight>;
using GradPtr = std::shared_ptr<Grad>;
using InputsShape = std::vector<std::shared_ptr<std::vector<size_t>>>;
using InputsShapePtr = std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>>;

constexpr size_t INDEX_NOT_SEND = UINT_MAX;
using OptimOriginIdx = std::map<std::string, size_t>;
using OptimPSSendIdx = std::map<std::string, size_t>;

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

#define EXC_IF_VEC_IDX_OOB(vec, idx)                                                            \
  {                                                                                             \
    size_t vec_size = vec.size();                                                               \
    if (idx >= vec_size) {                                                                      \
      MS_LOG(EXCEPTION) << "Vector " << #vec << " size is " << vec_size << ". So index " << idx \
                        << " is out of bound.";                                                 \
    }                                                                                           \
  }
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CONSTANTS_H_
