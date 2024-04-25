/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd

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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_SOLVER_PRE_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_SOLVER_PRE_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stack>
#include <vector>
#include <climits>
#include "utils/hash_map.h"
#include "include/backend/kernel_graph.h"

using mindspore::HashMap;
using std::vector;

namespace mindspore {
namespace somas {
constexpr char const *sortingNames[7] = {"size(>), index(<)",
                                         "can_reuse_mem(<), size(>), index(>)",
                                         "size(>), index(>)",
                                         "size(>), constraints(<), index(<)",
                                         "size(>), constraints(<), index(>)",
                                         "size(>), constraints(>), index(<)",
                                         "size(>), constraints(>), index(>)"};
constexpr char const *branchingNames[4] = {"bestfit", "smallest", "largest", "worstfit"};
constexpr char const *algorithmTypeNames[2] = {"Shared Objects", "Single Object"};
constexpr auto kParallelComputeSizeThreshold = 2000;
constexpr auto kHalfByteSize = 4;
enum Status { FAILED, SUCCESS };
enum AlgorithmType { kManyObjects = 0, kSingleObject, kNumAlgorithmTypes };
enum SortingType {
  kGreaterSizeSmallerIndex = 0,
  kSmallerReusePeakMemGreaterSizeSmallerIndex,
#ifdef SOMAS_DEBUG
  kGreaterSizeGreaterIndex,
  kGreaterSizeSmallerConstraintsSmallerIndex,
  kGreaterSizeSmallerConstraintsGreaterIndex,
  kGreaterSizeGreaterConstraintsSmallerIndex,
  kGreaterSizeGreaterConstraintsGreaterIndex,
#endif
  kNumSortingTypes
};
enum FittingType {
  kBest = 0,
  kSmallest,
#ifdef SOMAS_DEBUG
  kLargest,
  kWorst,
#endif
  kNumFittingTypes
};

struct BestInfo {
  size_t best_sol, worst, best, best_timing;
  AlgorithmType best_algo;
  BestInfo() : best_sol(0), worst(0), best(SIZE_MAX), best_timing(SIZE_MAX), best_algo(kManyObjects) {}
};

class DynamicBitSet {
  inline size_t GetIndex(size_t index) const { return index / bit_width_; }

  inline uint64_t GetBitMask(size_t index) const {
    return ((static_cast<uint64_t>(0x1)) << ((bit_width_ - 1) - (index % bit_width_)));
  }

  inline void Reset(uint64_t val) {
    bit_.clear();
    bit_.assign(bit_size_, val);
  }

 public:
  static constexpr size_t bit_width_ = 64;
  size_t bit_size_;
  std::vector<uint64_t> bit_;
  explicit DynamicBitSet(size_t count) : bit_size_((count + bit_width_ - 1) / bit_width_) { Reset(0x0); }

  ~DynamicBitSet() = default;

  DynamicBitSet(const DynamicBitSet &) = default;
  DynamicBitSet &operator=(const DynamicBitSet &) = default;

  void SetBitTrue(size_t index, bool log = false) {
    if (log) {
      MS_LOG(INFO) << GetIndex(index) << " " << GetBitMask(index);
    }
    bit_[GetIndex(index)] |= GetBitMask(index);
  }

  void SetBitFalse(size_t index) { bit_[GetIndex(index)] &= (~GetBitMask(index)); }

  bool IsBitTrue(size_t index) const { return (bit_[GetIndex(index)] & GetBitMask(index)) != 0x0; }

  size_t CountOnesNum() const {
    size_t ret = 0;
    static unsigned char ones_num_in_hex[] = "\0\1\1\2\1\2\2\3\1\2\2\3\2\3\3\4";
    for (size_t i = 0; i < bit_size_; i++) {
      auto value = bit_[i];
      if (value == 0) {
        continue;
      }
      auto *char_value = reinterpret_cast<unsigned char *>(&value);
      for (size_t j = 0; j < bit_width_ / CHAR_BIT; j++) {
        ret += ones_num_in_hex[static_cast<int>(char_value[j] & 0xF)];
        char_value[j] >>= kHalfByteSize;
        ret += ones_num_in_hex[static_cast<int>(char_value[j] & 0xF)];
      }
    }
    return ret;
  }

  void Log() {
    std::cout << "Start Print Bitset ";
    for (size_t i = 0; i < bit_size_; i++) {
      std::cout << " bit [" << std::dec << i << "] = " << std::hex << bit_[i] << std::dec;
    }
    std::cout << std::endl;
  }

  friend void Union(DynamicBitSet *a, DynamicBitSet *b) {
    for (size_t i = 0; i < (*a).bit_size_; i++) {
      (*a).bit_[i] |= (*b).bit_[i];
    }
  }

  friend void And(DynamicBitSet *a, DynamicBitSet *b) {
    for (size_t i = 0; i < (*a).bit_size_; i++) {
      (*a).bit_[i] &= (*b).bit_[i];
    }
  }
};

class VectorBitSet {
 public:
  explicit VectorBitSet(size_t count) : bit_(count, false), bit_size_(count) {}
  ~VectorBitSet() = default;

  VectorBitSet(const VectorBitSet &) = default;
  VectorBitSet &operator=(const VectorBitSet &) = default;

  void SetBitTrue(size_t index) { bit_[index] = true; }

  void SetBitFalse(size_t index) { bit_[index] = false; }

  void SetBit(size_t index, bool value) { bit_[index] = value; }

  bool IsBitTrue(size_t index) const { return bit_[index]; }

  std::vector<bool> bit_;
  size_t bit_size_;
};

struct SomasSolverTensorDesc {
  size_t index_;
  size_t size_;
  size_t offset_;
  bool lifelong_;
  size_t can_reuse_peak_mem_;
  bool is_graph_output_;
  using SomasSolverTensorDescPtr = std::shared_ptr<SomasSolverTensorDesc>;
  SomasSolverTensorDescPtr right_;
  SomasSolverTensorDescPtr left_;
  bool blocked_;

  SomasSolverTensorDesc() = default;

  SomasSolverTensorDesc(size_t index, size_t size, size_t offset, bool blifelong)
      : index_(index),
        size_(size),
        offset_(offset),
        lifelong_(blifelong),
        can_reuse_peak_mem_(0),
        is_graph_output_(false),
        right_(nullptr),
        left_(nullptr),
        blocked_(false) {}

  void Update(size_t index, size_t size, size_t offset, size_t can_reuse_peak_mem, bool is_graph_output,
              bool blifelong) {
    index_ = index;
    size_ = size;
    offset_ = offset;
    lifelong_ = blifelong;
    can_reuse_peak_mem_ = can_reuse_peak_mem;
    is_graph_output_ = is_graph_output;
  }

  friend std::ostream &operator<<(std::ostream &out, const SomasSolverTensorDescPtr n) {
    out << n->index_ << " " << n->size_ << " " << n->offset_ << "\n";
    return out;
  }
  friend std::istream &operator>>(std::istream &in, const SomasSolverTensorDescPtr &n) {
    in >> n->index_ >> n->size_ >> n->offset_;
    return in;
  }
};
using SomasSolverTensorDescPtr = std::shared_ptr<SomasSolverTensorDesc>;
typedef mindspore::HashMap<size_t, SomasSolverTensorDescPtr> TensorsDescMap;
class SomasSolverPre {
 public:
  SomasSolverPre() = default;
  ~SomasSolverPre() = default;

  SomasSolverPre(const SomasSolverPre &) = delete;
  SomasSolverPre &operator=(const SomasSolverPre &) = delete;

  size_t GetMaxOffset() const { return max_offset_; }

  Status Solving(const session::KernelGraph &graph, TensorsDescMap *ptensors,
                 const std::vector<VectorBitSet> *pConstraints, const vector<vector<size_t>> &continuous_v,
                 const std::vector<int> &core_list,
                 bool bVerifySolution,  // true -> Check continuous and non overlapping constraints solution
                 bool ball = true,      // true -> run full set of heuristics, false -> run single heuristic specified
                 SortingType sorting = kGreaterSizeSmallerIndex, FittingType fitting = kBest,
                 AlgorithmType algorithm = kManyObjects);

  void Log(const session::KernelGraph &graph, const TensorsDescMap &tensors,
           const std::vector<VectorBitSet> *pConstraints, const vector<vector<size_t>> &continuous_v) const;

  Status CheckTensors(const TensorsDescMap *pTensors, uint32_t index1, uint32_t index2) const;
  Status AddContiguousInfoInMap(const vector<vector<size_t>> &continuous_v, TensorsDescMap *pTensors) const;
  Status AddContiguousInfoInMultiMaps(const vector<vector<size_t>> &continuous_v, vector<TensorsDescMap> *vecTensorsMap,
                                      const TensorsDescMap *pTensors) const;

 private:
  size_t max_offset_;
  void SolverInputLog(const session::KernelGraph &graph, const TensorsDescMap &tensors,
                      const vector<vector<size_t>> &continuous_v) const;
  void SolverOutputLog(const session::KernelGraph &graph, const TensorsDescMap &tensors) const;
  vector<TensorsDescMap> CreateTensorsMaps(const TensorsDescMap &tensors, size_t total_sol) const;
  void TensorRelationLog(const std::vector<VectorBitSet> *pConstraints, const session::KernelGraph &graph) const;
};
using SomasSolverPrePtr = std::shared_ptr<SomasSolverPre>;
}  // namespace somas
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_SOLVER_PRE_H_
