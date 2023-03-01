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

#ifndef PARALLEL_AUTO_PARALLEL_REC_COST_H_
#define PARALLEL_AUTO_PARALLEL_REC_COST_H_

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "frontend/parallel/auto_parallel/rec_core/rec_graph.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_strategy.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace parallel {
#define DOUBLE_MAX (std::numeric_limits<double>::max)()
#define DOUBLE_LOWEST (std::numeric_limits<double>::lowest)()
#define DOUBLE_MIN (std::numeric_limits<double>::min)()

constexpr double MATMUL_MEM_COEF = 0.25;
constexpr size_t REDIS_COEF = 16;

double CostRedis(const Graph::NodeType &node,
                 const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                 const std::vector<std::vector<float>> &mode, const Graph &graph);

double CostRedisWithAdjacentNode(const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                 const std::vector<std::vector<float>> &mode, size_t i_strategy, size_t i_node,
                                 double tensor_size, bool is_search_forward);

// class CostMatMul is used to compute the cost of MatMul operator.
class CostMatMul {
 public:
  StrategyRec GetOptimalStr(const Graph::NodeType &node,
                            const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                            const Graph &graph, const bool isTraining);

  double GetMaxCostIn(const OperatorRec &op);

 private:
  double StrConcatDimI(int64_t a, int64_t b) {
    cost_in_i_ = (static_cast<double>(a) * static_cast<double>(b)) / 2.0;
    cost_in_i_ = cost_in_i_ * MATMUL_MEM_COEF;

    return cost_in_i_;
  }

  double StrConcatDimJ(int64_t a, int64_t b) {
    cost_in_j_ = (static_cast<double>(a) * static_cast<double>(b)) / 2.0;

    return cost_in_j_;
  }

  double StrReduceDimK(int64_t a, int64_t b) {
    cost_in_k_ = (static_cast<double>(a) * static_cast<double>(b)) / 2.0;

    return cost_in_k_;
  }

  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str) const;

  double cost_in_i_ = 0;

  double cost_in_j_ = 0;

  double cost_in_k_ = 0;
};  // class CostMatMul is used to compute the cost of MatMul operator.

// class CostConvolution is used to compute the cost of Conv operator.
class CostConvolution {
 public:
  StrategyRec GetOptimalStr(const Graph::NodeType &node,
                            const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                            const Graph &graph, bool channel_partition);

  double GetMinCostIn(const Graph::NodeType &node);

 private:
  double StrDimB(int64_t TensorFilter) {
    cost_in_b_ = static_cast<double>((TensorFilter) / 2.0);

    return cost_in_b_;
  }

  double StrDimI(int64_t TensorIn, int64_t TensorFilter) {
    cost_in_i_ = static_cast<double>((TensorIn + TensorFilter) / 2.0);

    return cost_in_i_;
  }

  double StrDimJ(int64_t TensorIn, int64_t TensorFilter) {
    cost_in_j_ = static_cast<double>((TensorIn + TensorFilter) / 2.0);

    return cost_in_j_;
  }

  double StrDimK(int64_t TensorIn) {
    cost_in_k_ = static_cast<double>((TensorIn) / 2.0);

    return cost_in_k_;
  }

  double StrDimDI(int64_t TensorIn, int64_t TensorOut) {
    cost_in_di_ = static_cast<double>((TensorIn + TensorOut) / 2.0);

    return cost_in_di_;
  }

  double StrDimDJ(int64_t TensorIn, int64_t TensorOut) {
    cost_in_dj_ = static_cast<double>((TensorIn + TensorOut) / 2.0);

    return cost_in_dj_;
  }

  double StrDimQ(int64_t TensorOut) {
    cost_in_q_ = static_cast<double>((TensorOut) / 2.0);

    return cost_in_q_;
  }

  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str) const;

  double cost_in_b_ = 0;

  double cost_in_i_ = 0;

  double cost_in_j_ = 0;

  double cost_in_k_ = 0;

  double cost_in_di_ = 0;

  double cost_in_dj_ = 0;

  double cost_in_q_ = 0;
};  // class CostConvolution is used to compute the cost of Conv operator.

// class CostPooling is used to compute the cost of Pooling operator.
class CostPooling {
 public:
  StrategyRec GetOptimalStr(const Graph::NodeType &node,
                            const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                            const Graph &graph) const;

  double GetMinCostIn() const { return cost_in_; }

 private:
  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str) const;

  double cost_in_ = 0;
};  // class CostPooling is used to compute the cost of Pooling operator.

// class CostReshape is used to compute the cost of Reshape operator.
class CostReshape {
 public:
  StrategyRec GetOptimalStr(const Graph::NodeType &node) const;

  double GetMinCostIn() const { return cost_in_; }

 private:
  StrategyRec ChoseStr(StrategyRec str) const;

  double cost_in_ = 0;
};  // class CostReshape is used to compute the cost of Reshape operator.

// class CostCommon is used to compute the cost of an element-wise operator
class CostCommon {
 public:
  virtual ~CostCommon() = default;

  virtual StrategyRec GetOptimalStr(const Graph::NodeType &node,
                                    const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                    const Graph &graph);

  virtual double GetMinCostIn() const { return cost_in_; }

 protected:
  virtual StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str);

  double cost_in_ = 0;
};  // class CostCommon is used to compute the cost of an element-wise operator

// class CostBiasAdd is used to compute the cost of the addition between a tensor and a bias
class CostBiasAdd : public CostCommon {
 protected:
  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str) override;
};

// class CostAdd is used to compute the cost of Add operator.
class CostTensorAdd : public CostCommon {
 protected:
  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str) override;
};

// all the following operation are element-wise and have the same cost
class CostReLU : public CostCommon {};
class CostLog : public CostCommon {};
class CostExp : public CostCommon {};
class CostAdd : public CostCommon {};
class CostSub : public CostCommon {};
class CostMul : public CostCommon {};
class CostDiv : public CostCommon {};
class CostSqueeze : public CostCommon {};
class CostCast : public CostCommon {};

// class BatchParallel is used to compute the cost of BatchParallel operator.
class CostBatchParallel {
 public:
  virtual ~CostBatchParallel() = default;

  virtual StrategyRec GetOptimalStr(const Graph::NodeType &node);

  virtual double GetMaxCostIn() const { return DOUBLE_MAX; }

 protected:
  virtual StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str);

  double cost_in_ = 0;
};  // class BatchParallel is used to compute the cost of BatchParallel operator.

class CostBatchNorm : public CostBatchParallel {};
class CostOneHot : public CostBatchParallel {};
class CostPRelu : public CostBatchParallel {};
class CostSoftmax : public CostBatchParallel {};

class CostSoftmaxCrossEntropyWithLogits : public CostBatchParallel {
 protected:
  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str) override;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_COST_H_
