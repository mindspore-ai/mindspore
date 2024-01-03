/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PRIME_GENERATOR_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PRIME_GENERATOR_H_
#include <iostream>
#include <cstdio>
#include <vector>
#include <chrono>
#include <memory>
#include <set>
#include <map>
#include "mindspore/core/base/base.h"
#include "mindspore/core/ir/func_graph.h"

namespace mindspore::parallel {
using Shape = std::vector<int64_t>;
using AssembledDynamicDimsMapping = std::map<int64_t, AnfNodePtr>;

class PrimeGenerator {
 public:
  static PrimeGenerator *GetInstance() {
    static PrimeGenerator m_instance;
    return &m_instance;
  }
  PrimeGenerator(const PrimeGenerator &) = delete;
  const PrimeGenerator &operator=(const PrimeGenerator &) = delete;
  int64_t GetCoprimeNum(const Shape &tensor_shape);

 private:
  PrimeGenerator();
  ~PrimeGenerator() = default;

  std::vector<int64_t> prime_table_;
};

class DecomposeDim {
 public:
  DecomposeDim() = default;

  void AppendPrimeDim(int64_t v, size_t index) {
    this->prime_.emplace_back(v);
    this->index_.emplace_back(index);
  }

  static DecomposeDim Decompose(int64_t dim, const std::vector<int64_t> &src_factor);

  std::vector<int64_t> prime() const { return this->prime_; }
  std::vector<int64_t> prime_index() const { return this->index_; }
  int64_t factor() const { return this->factor_; }
  void set_factor(int64_t f) { this->factor_ = f; }
  bool IsDecomposable() const { return !this->prime_.empty(); }
  bool CreateShapeTracingGraph(const AssembledDynamicDimsMapping &mapping, const FuncGraphPtr &func_graph,
                               std::vector<AnfNodePtr> *anf_nodes);

 private:
  std::vector<int64_t> prime_;
  std::vector<int64_t> index_;
  int64_t factor_ = 1;
};
// DecomposeDim MatchDims(int64_t dim, const std::vector<int64_t> &src_factor);
}  // namespace mindspore::parallel
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PRIME_GENERATOR_H_
