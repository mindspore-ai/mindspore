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

#include "parallel/auto_parallel/rec_core/rec_graph.h"
#include "parallel/auto_parallel/rec_core/rec_strategy.h"

namespace mindspore {
namespace parallel {
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
                            const Graph &graph);

  double GetMinCostIn(const OperatorRec &op);

 private:
  double StrConcatDimI(int32_t a, int32_t b) {
    cost_in_i_ = (static_cast<double>(a) * static_cast<double>(b)) / 2.0;

    return cost_in_i_;
  }

  double StrConcatDimJ(int32_t a, int32_t b) {
    cost_in_j_ = (static_cast<double>(a) * static_cast<double>(b)) / 2.0;

    return cost_in_j_;
  }

  double StrReduceDimK(int32_t a, int32_t b) {
    cost_in_k_ = (static_cast<double>(a) * static_cast<double>(b)) / 2.0;

    return cost_in_k_;
  }

  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str);

  double cost_in_i_ = 0;

  double cost_in_j_ = 0;

  double cost_in_k_ = 0;
};  // class CostMatMul is used to compute the cost of MatMul operator.

// class CostConvolution is used to compute the cost of Conv operator.
class CostConvolution {
 public:
  StrategyRec GetOptimalStr(const Graph::NodeType &node,
                            const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                            const Graph &graph);

  double GetMinCostIn(const Graph::NodeType &node);

 private:
  double StrDimB(int32_t TensorFilter) {
    cost_in_b_ = static_cast<double>((TensorFilter) / 2.0);

    return cost_in_b_;
  }

  double StrDimI(int32_t TensorIn, int32_t TensorFilter) {
    cost_in_i_ = static_cast<double>((TensorIn + TensorFilter) / 2.0);

    return cost_in_i_;
  }

  double StrDimJ(int32_t TensorIn, int32_t TensorFilter) {
    cost_in_j_ = static_cast<double>((TensorIn + TensorFilter) / 2.0);

    return cost_in_j_;
  }

  double StrDimK(int32_t TensorIn) {
    cost_in_k_ = static_cast<double>((TensorIn) / 2.0);

    return cost_in_k_;
  }

  double StrDimDI(int32_t TensorIn, int32_t TensorOut) {
    cost_in_di_ = static_cast<double>((TensorIn + TensorOut) / 2.0);

    return cost_in_di_;
  }

  double StrDimDJ(int32_t TensorIn, int32_t TensorOut) {
    cost_in_dj_ = static_cast<double>((TensorIn + TensorOut) / 2.0);

    return cost_in_dj_;
  }

  double StrDimQ(int32_t TensorOut) {
    cost_in_q_ = static_cast<double>((TensorOut) / 2.0);

    return cost_in_q_;
  }

  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str);

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
                            const Graph &graph);

  double GetMinCostIn() const { return cost_in_; }

 private:
  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str);

  double cost_in_ = 0;
};  // class CostPooling is used to compute the cost of Pooling operator.

// class CostAdd is used to compute the cost of Add operator.
class CostAdd {
 public:
  StrategyRec GetOptimalStr(const Graph::NodeType &node,
                            const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                            const Graph &graph);

  double GetMinCostIn() const { return cost_in_; }

 private:
  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str);

  double cost_in_ = 0;
};  // class CostAdd is used to compute the cost of Add operator.

// class CostReshape is used to compute the cost of Reshape operator.
class CostReshape {
 public:
  StrategyRec GetOptimalStr(const Graph::NodeType &node) const;

  double GetMinCostIn() const { return cost_in_; }

 private:
  StrategyRec ChoseStr(StrategyRec str) const;

  double cost_in_ = 0;
};  // class CostReshape is used to compute the cost of Reshape operator.

// class CostBiasAdd is used to compute the cost of BiasAdd operator.
class CostBiasAdd {
 public:
  StrategyRec GetOptimalStr(const Graph::NodeType &node,
                            const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                            const Graph &graph);

  double GetMinCostIn() const { return cost_in_; }

 private:
  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str);

  double cost_in_ = 0;
};  // class CostBiasAdd is used to compute the cost of BiasAdd operator.

// class CostCommon is used to compute the cost of the element independent operator.
class CostCommon {
 public:
  StrategyRec GetOptimalStr(const Graph::NodeType &node,
                            const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                            const Graph &graph);

  double GetMinCostIn() const { return cost_in_; }

 private:
  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str);

  double cost_in_ = 0;
};  // class CostCommon is used to compute the cost of Softmax & || Activation operator.

// class BatchNorm is used to compute the cost of BatchNorm operator.
class CostBatchNorm {
 public:
  StrategyRec GetOptimalStr(const Graph::NodeType &node,
                            const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                            const Graph &graph);

  double GetMinCostIn() const { return 0.0; }

 private:
  double StrDimB(int32_t Tensor) {
    cost_in_b_ = (static_cast<double>(Tensor) * 4.0) / 2.0;

    return cost_in_b_;
  }

  double StrDimC() {
    cost_in_c_ = 0.0;

    return cost_in_c_;
  }

  double StrDimH(int32_t Tensor) {
    cost_in_h_ = (static_cast<double>(Tensor) * 4.0) / 2.0;

    return cost_in_h_;
  }

  double StrDimW(int32_t Tensor) {
    cost_in_w_ = (static_cast<double>(Tensor) * 4.0) / 2.0;

    return cost_in_w_;
  }

  StrategyRec ChoseStr(const std::vector<double> &cost_op, StrategyRec str);

  double cost_in_b_ = 0;

  double cost_in_c_ = 0;

  double cost_in_h_ = 0;

  double cost_in_w_ = 0;
};  // class BatchNorm is used to compute the cost of BatchNorm operator.
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_COST_H_
