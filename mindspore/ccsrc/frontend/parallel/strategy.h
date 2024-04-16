/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"

namespace mindspore {
namespace parallel {
constexpr int MIN_SLICE_NUM = 1;

using Dimensions = Shape;
using NewDimensions = ShapeBasePtr;
using Strategies = std::vector<Dimensions>;
using NewStrategies = std::vector<NewDimensions>;
class Strategy;
using StrategyPtr = std::shared_ptr<Strategy>;

class Strategy {
 public:
  Strategy(int64_t stage, Strategies inputs)
      : stage_(stage), inputs_(std::move(inputs)), internal_size_(0), internal_stragies_() {}
  Strategy(int64_t stage, NewStrategies inputs)
      : stage_(stage), inputs_new_(std::move(inputs)), internal_size_(0), internal_stragies_() {}

  Strategy(const Strategy &another_stra) : stage_(another_stra.GetInputStage()) {
    inputs_ = another_stra.GetInputDim();
    inputs_new_ = another_stra.GetInputNewDim();
    internal_size_ = another_stra.GetInternalSize();
    if (internal_size_ != 0) {
      internal_stragies_ = another_stra.GetInternalStrategies();
    } else {
      internal_stragies_ = {};
    }
  }

  Strategy &operator=(const Strategy &another_stra) {
    if (this != &another_stra) {
      inputs_ = another_stra.GetInputDim();
      inputs_new_ = another_stra.GetInputNewDim();
      internal_size_ = another_stra.GetInternalSize();
      if (internal_size_ != 0) {
        internal_stragies_ = another_stra.GetInternalStrategies();
      } else {
        internal_stragies_ = {};
      }
    }
    return *this;
  }

  ~Strategy() = default;
  size_t GetInputNumber() const {
    if (inputs_new_.empty()) {
      return inputs_.size();
    } else {
      return inputs_new_.size();
    }
  }
  bool HasTupleInTupleStrategy() const { return !inputs_new_.empty(); }
  Strategies GetInputDim() const { return inputs_; }
  NewStrategies GetInputNewDim() const { return inputs_new_; }
  int64_t GetInputStage() const { return stage_; }
  void ExpandInputDimFromOneToTwo() {
    if (inputs_new_.empty()) {
      if (inputs_.size() == 1) {
        inputs_.push_back(inputs_[0]);
      }
    } else {
      if (inputs_new_.size() == 1) {
        inputs_new_.push_back(inputs_new_[0]);
      }
    }
  }
  void ResetInputs(const Strategies &input) { inputs_ = input; }
  void ResetInputs(const NewStrategies &input) { inputs_new_ = input; }
  std::vector<StrategyPtr> GetInternalStrategies() const { return internal_stragies_; }
  size_t GetInternalSize() const { return internal_size_; }

  bool IsEqual(const StrategyPtr &another_stra) {
    if (another_stra == nullptr) {
      return false;
    }

    std::vector<Dimensions> squashed_inputs_stra;
    std::vector<size_t> stra_size;
    std::vector<Dimensions> in_squashed_inputs_stra;
    std::vector<size_t> in_stra_size;
    // Current stra is tuple in tuple or not
    std::tie(squashed_inputs_stra, stra_size) = GetSquashedStraAndSize();
    // Input stra is tuple in tuple or not
    std::tie(in_squashed_inputs_stra, in_stra_size) = GetInSquashedStraAndSize(another_stra);
    if ((stage_ != another_stra->GetInputStage()) || (squashed_inputs_stra != in_squashed_inputs_stra) ||
        (stra_size != in_stra_size)) {
      return false;
    }

    return true;
  }

  int64_t PartitionNum() {
    int64_t divergence = 1;
    if (inputs_new_.empty()) {
      for (size_t i = 0; i < inputs_.size(); ++i) {
        for (size_t j = 0; j < inputs_[i].size(); ++j) {
          divergence *= inputs_[i][j];
        }
      }
    } else {
      for (const auto &stra : inputs_new_) {
        ObtainPartionNum(stra, &divergence);
      }
    }
    return divergence;
  }

  bool Compare(const StrategyPtr &another_stra) {
    if (this->PartitionNum() > another_stra->PartitionNum()) {
      return true;
    } else if (this->PartitionNum() < another_stra->PartitionNum()) {
      return false;
    }
    std::vector<Dimensions> squashed_inputs_stra;
    std::vector<size_t> stra_size;
    std::vector<Dimensions> in_squashed_inputs_stra;
    std::vector<size_t> in_stra_size;
    std::tie(squashed_inputs_stra, stra_size) = GetSquashedStraAndSize();
    std::tie(in_squashed_inputs_stra, in_stra_size) = GetInSquashedStraAndSize(another_stra);
    return squashed_inputs_stra > in_squashed_inputs_stra;
  }

  // Include 'another_stra' into this strategy
  void CoverStrategy(const StrategyPtr &another_stra) {
    internal_stragies_.push_back(another_stra);
    internal_size_++;
  }

  std::string ToString() const {
    std::ostringstream oss;
    oss << "[";
    if (this->HasTupleInTupleStrategy()) {
      for (size_t i = 0; i < this->GetInputNumber(); ++i) {
        CovertStrategyToString(this->GetInputNewDim()[i], &oss);
        if (i != this->GetInputNumber() - 1) {
          oss << ", ";
        }
      }
    } else {
      for (size_t i = 0; i < this->GetInputNumber(); ++i) {
        oss << "[";
        for (size_t j = 0; j < this->GetInputDim()[i].size(); ++j) {
          oss << std::to_string(this->GetInputDim()[i][j]);
          if (j != this->GetInputDim()[i].size() - 1) {
            oss << ", ";
          }
        }
        oss << "]";
        if (i != this->GetInputNumber() - 1) {
          oss << ", ";
        }
      }
    }
    oss << "]";
    return oss.str();
  }

 private:
  const int64_t stage_;

  // The size of Dimensions must be equal to inputs_ tensor dimension.
  Strategies inputs_;
  NewStrategies inputs_new_;
  size_t internal_size_ = 0;
  std::vector<StrategyPtr> internal_stragies_;

  void ObtainPartionNum(const NewDimensions &inputs_stra, int64_t *divergence) {
    if (inputs_stra->is_list()) {
      for (size_t i = 0; i < inputs_stra->size(); ++i) {
        ObtainPartionNum(inputs_stra->GetElement(SizeToLong(i)), divergence);
      }
    } else {
      auto stra = inputs_stra->GetValue();
      for (const auto &stra_value : stra) {
        *divergence *= stra_value;
      }
    }
  }

  std::pair<std::vector<Dimensions>, std::vector<size_t>> GetInSquashedStraAndSize(const StrategyPtr &inputs_stra) {
    std::vector<Dimensions> in_squashed_inputs_stra;
    std::vector<size_t> in_stra_size;
    if (inputs_stra->HasTupleInTupleStrategy()) {
      auto local_stra = inputs_stra->GetInputNewDim();
      for (const auto &stra : local_stra) {
        auto all_stra = stra->GetAllElements();
        in_squashed_inputs_stra.insert(in_squashed_inputs_stra.end(), all_stra.begin(), all_stra.end());
        in_stra_size.push_back(stra->size());
      }
    } else {
      in_squashed_inputs_stra = inputs_stra->GetInputDim();
      for (size_t i = 0; i < in_squashed_inputs_stra.size(); ++i) {
        in_stra_size.push_back(in_squashed_inputs_stra[i].size());
      }
    }
    return std::make_pair(in_squashed_inputs_stra, in_stra_size);
  }

  std::pair<std::vector<Dimensions>, std::vector<size_t>> GetSquashedStraAndSize() {
    std::vector<Dimensions> squashed_inputs_stra;
    std::vector<size_t> stra_size;
    if (inputs_new_.empty()) {
      squashed_inputs_stra = inputs_;
      for (size_t i = 0; i < inputs_.size(); ++i) {
        stra_size.push_back(inputs_[i].size());
      }
    } else {
      for (const auto &stra : inputs_new_) {
        auto all_stra = stra->GetAllElements();
        squashed_inputs_stra.insert(squashed_inputs_stra.end(), all_stra.begin(), all_stra.end());
        stra_size.push_back(stra->size());
      }
    }
    return std::make_pair(squashed_inputs_stra, stra_size);
  }

  void CovertStrategyToString(const NewDimensions &stra, std::ostringstream *oss) const {
    *oss << "[";
    if (stra->is_list()) {
      for (size_t i = 0; i < stra->size(); ++i) {
        CovertStrategyToString(stra->GetElement(SizeToLong(i)), oss);
        if (i != stra->size() - 1) {
          *oss << ", ";
        }
      }
    } else {
      auto stra_value = stra->GetValue();
      for (size_t i = 0; i < stra_value.size(); ++i) {
        *oss << stra_value[i];
        if (i != stra_value.size() - 1) {
          *oss << ", ";
        }
      }
    }
    *oss << "]";
  }
};

inline StrategyPtr NewStrategy(const int64_t stage, const Strategies &inputs) {
  return std::make_shared<Strategy>(stage, inputs);
}
inline StrategyPtr NewStrategy(const int64_t stage, const NewStrategies &inputs) {
  return std::make_shared<Strategy>(stage, inputs);
}
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_H_
