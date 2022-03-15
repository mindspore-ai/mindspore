/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_KERNEL_COMMON_UTILS_H_

#include <dirent.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <utility>
#include <nlohmann/json.hpp>
#include "include/common/utils/utils.h"
#include "kernel/kernel.h"
#include "kernel/oplib/opinfo.h"
#include "kernel/kernel_build_info.h"

namespace mindspore {
namespace kernel {
constexpr auto kAkgKernelMeta = "kernel_meta/";
constexpr auto kProcessorAiCore = "aicore";
constexpr auto kProcessorAiCpu = "aicpu";
constexpr auto kProcessorCuda = "cuda";
constexpr auto kProcessorCpu = "cpu";
constexpr auto kProcessorUnknown = "unknown";
constexpr auto kJsonSuffix = ".json";
constexpr auto kInfoSuffix = ".info";
constexpr unsigned int AUTODIFF_COMPILE_OVERTIME = 600;

const std::vector<std::string> support_devices = {"aicore", "aicpu", "cuda"};

// an enum to indicate a vector or matrix alignment direction.
// real_data: [1,2,3] left_align: [1,2,3,0] right_align:[0,1,2,3]
namespace MatrixDiag {
enum Alignment { RIGHT = 0, LEFT = 1 };
static const mindspore::HashMap<std::string, std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment>> AlignmentMap{
  {"RIGHT_LEFT", {MatrixDiag::RIGHT, MatrixDiag::LEFT}},
  {"LEFT_RIGHT", {MatrixDiag::LEFT, MatrixDiag::RIGHT}},
  {"RIGHT_RIGHT", {MatrixDiag::RIGHT, MatrixDiag::RIGHT}},
  {"LEFT_LEFT", {MatrixDiag::LEFT, MatrixDiag::LEFT}}};
}  // namespace MatrixDiag

struct KernelMetaInfo {
  uintptr_t func_stub_;
  uint32_t block_dim_;
};
using KernelMetaPtr = std::shared_ptr<KernelMetaInfo>;

class KernelMeta {
 public:
  KernelMeta() = default;
  void Initialize();
  std::string Search(const std::string &kernel_name) const;
  bool Insert(const std::string &kernel_name, const std::string &kernel_json);
  std::string kernel_meta_path() const { return kernel_meta_path_; }
  bool initialized() const { return initialized_; }
  static KernelMeta *GetInstance() {
    static KernelMeta kernel_meta;
    return &kernel_meta;
  }
  ~KernelMeta() = default;

 private:
  bool initialized_ = false;
  std::string kernel_meta_path_;
  std::unordered_map<std::string, std::string> kernel_meta_map_;
};

class MatrixInfo {
 public:
  explicit MatrixInfo(size_t max_index, const std::vector<size_t> &matrix_shapes)
      : max_index_(max_index), shapes_(matrix_shapes) {
    current_indexes_.resize(shapes_.size(), 0);
  }
  ~MatrixInfo() = default;
  bool SetIndex(size_t start, size_t end) {
    // check data from start to end whether valid.
    if (start < min_index || end > max_index_ || start >= end) {
      return false;
    }
    // initial current indexes.
    int last_rank = SizeToInt(current_indexes_.size()) - 1;
    for (int i = last_rank; start != 0 && i >= 0; --i) {
      size_t position = IntToSize(i);
      current_indexes_[position] = start % shapes_.at(position);
      start = start / shapes_.at(position);
    }
    return true;
  }
  std::vector<size_t> IndexIterator() {
    if (is_first_iterator_) {
      is_first_iterator_ = false;
      return current_indexes_;
    }
    size_t last_rank = current_indexes_.size() - 1;
    current_indexes_[last_rank]++;
    for (size_t i = last_rank; current_indexes_.at(i) >= shapes_.at(i) && i > 0; --i) {
      current_indexes_[i] = 0;
      current_indexes_[i - 1] += 1;
    }
    is_first_iterator_ = false;
    return current_indexes_;
  }

 private:
  bool is_first_iterator_{true};
  size_t min_index{0};
  size_t max_index_{1};
  std::vector<size_t> shapes_;
  std::vector<size_t> current_indexes_;
};
using MatrixInfoPtr = std::shared_ptr<MatrixInfo>;

std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> GetAlignments(const std::string &alignment);
int CalDiagOffset(int diag_index, int max_diag_len, int inner_rows, int inner_cols,
                  const std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> &alignment);
std::string GetCompilerCachePath();
bool CheckCache(const std::string &kernel_name);
KernelPackPtr SearchCache(const std::string &kernel_name, const std::string &processor);
KernelPackPtr InsertCache(const std::string &kernel_name, const std::string &processor);
TypeId DtypeToTypeId(const std::string &dtypes);
std::string Dtype2ShortType(const std::string &dtypes);
size_t GetDtypeNbyte(const std::string &dtypes);
bool GetShapeSize(const std::vector<size_t> &shape, const TypePtr &type_ptr, int64_t *size_i);
bool ParseMetadata(const CNodePtr &kernel_node, const std::shared_ptr<const OpInfo> &op_info_ptr, Processor processor,
                   std::vector<std::shared_ptr<KernelBuildInfo>> *const kernel_info_list);
void SaveJsonInfo(const std::string &json_name, const std::string &info, const std::string &base_path);
std::string GetProcessor(const AnfNodePtr &anf_node);
Processor GetProcessor(const string &processor);
bool IsSameShape(const std::vector<size_t> &shape_a, const std::vector<size_t> &shape_b);
std::vector<std::pair<AnfNodePtr, size_t>> GetOutputIndex(const std::vector<AnfNodePtr> &node_list,
                                                          const std::vector<AnfNodePtr> &input_list,
                                                          const std::vector<AnfNodePtr> &output_list);
void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list);
void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list,
                         std::vector<AnfNodePtr> *input_list, std::vector<AnfNodePtr> *output_list);
void GetFuncGraphOutputNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *output_list);
void GetGraphRealOutput(const FuncGraphPtr &func_graph, std::vector<std::pair<AnfNodePtr, size_t>> *node_list);
bool IsWeightBoundary(const AnfNodePtr &node);
std::vector<int64_t> GetReduceAttrAxis(const CNodePtr &cnode);
std::string GetProcessorStr(const AnfNodePtr &anf_node);
Processor GetProcessorFromContext();
std::string GetStrProcessorFromContext();
float Scaling(size_t in_size, size_t out_size, bool align_corners);
float ScaleGrid(const int x, const float scale);
FusionType GetFusionTypeByName(const std::string &name);
std::string GetFusionNameByType(const kernel::FusionType &type);
std::vector<bool> Dec2Bin(const int64_t &mask);
void FillEmptyDims(const CNodePtr &kernel_node, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                   std::vector<int64_t> *stride, std::vector<size_t> *input_shape);
void ParseStrideSliceMasks(const CNodePtr &kernel_node, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                           std::vector<int64_t> *stride, const std::vector<size_t> &input_shape);
struct CachedInterpolation {
  size_t lower;
  size_t upper;
  float lerp;
};

void ComputeInterpolationWeights(const size_t out_size, const size_t in_size, const float scale,
                                 CachedInterpolation *interpolation);

template <typename T>
inline std::string Vector2Str(const std::vector<T> &inputs) {
  if (!inputs.empty()) {
    std::ostringstream oss;
    (void)std::copy(inputs.begin(), inputs.end() - 1, std::ostream_iterator<T>(oss, ", "));
    oss << inputs.back();
    return oss.str();
  }
  return "";
}

template <typename T>
inline std::string Map2Str(const std::map<std::string, T> value) {
  std::stringstream ss;
  ss << "(";
  for (auto it = value.begin(); it != value.end(); it++) {
    if (it == value.begin()) {
      ss << it->first;
    } else {
      ss << ", " << it->first;
    }
  }
  ss << ")";
  return ss.str();
}

template <typename T>
inline std::string Unorderedmap2Str(const std::unordered_map<std::string, T> value) {
  std::stringstream ss;
  ss << "(";
  for (auto it = value.begin(); it != value.end(); it++) {
    if (it == value.begin()) {
      ss << it->first;
    } else {
      ss << ", " << it->first;
    }
  }
  ss << ")";
  return ss.str();
}

template <typename T>
inline T ComputeLerp(T top_left, T top_right, T bottom_left, T bottom_right, T x_lerp, T y_lerp) {
  T top = top_left + (top_right - top_left) * x_lerp;
  T bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

void CastShapeSizeToLong(const std::vector<size_t> &shape, std::vector<int64_t> *long_shape);
void CheckSliceValid(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                     const std::vector<int64_t> &step, const std::vector<int64_t> &input_shape);
size_t CalOffset(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                 const std::vector<int64_t> &dim_offset);
std::vector<int64_t> CalDimOffset(const std::vector<int64_t> &input_shape);
size_t GetCopySize(const std::vector<int64_t> &dim_offset, const std::vector<int64_t> &start,
                   const std::vector<int64_t> &stop);
size_t UnitSizeInBytes(const mindspore::TypeId &t);

class KernelAttr {
 public:
  using DataType = std::pair<TypeId, std::string>;
  KernelAttr() = default;
  ~KernelAttr() = default;

  KernelAttr &AddInputAttr(const TypeId &ms_type, const std::string &format = kOpFormat_DEFAULT);
  KernelAttr &AddOutputAttr(const TypeId &ms_type, const std::string &format = kOpFormat_DEFAULT);
  KernelAttr &AddAllSameAttr(const bool &all_same);
  KernelAttr &AddOutInRef(size_t output_index, size_t input_index);

  const DataType &GetInputAttr(const size_t index) const { return input_type_[index]; }
  const DataType &GetOutputAttr(const size_t index) const { return output_type_[index]; }
  const bool &GetAllSame() const { return all_same_; }

  size_t GetInputSize() const { return input_type_.size(); }
  size_t GetOutputSize() const { return output_type_.size(); }
  const OutputInputRefMap &GetOutInRefMap() const { return out_in_ref_map_; }

  void SetInputAttrList(const std::vector<DataType> &addr_list);

 private:
  std::vector<DataType> input_type_;
  std::vector<DataType> output_type_;
  bool all_same_{false};

  // The map between kernel's output and input ref relationship.
  OutputInputRefMap out_in_ref_map_;
};
std::ostream &operator<<(std::ostream &os, KernelAttr kernel_attr);

std::pair<bool, size_t> MatchKernelAttr(const KernelAttr &kernel_attr, const std::vector<KernelAttr> &attr_list);
KernelAttr GetKernelAttrFromBuildInfo(const KernelBuildInfoPtr &build_info);
KernelAttr GetKernelAttrFromNode(const AnfNodePtr &kernel_node);

#define CHECK_KERNEL_INPUTS_NUM(actual_inputs_num, expect_inputs_num, kernel_name)                     \
  do {                                                                                                 \
    if ((actual_inputs_num) != (expect_inputs_num)) {                                                  \
      MS_LOG(EXCEPTION) << (kernel_name) << " requires " << (expect_inputs_num) << " inputs, but got " \
                        << (actual_inputs_num) << ".";                                                 \
    }                                                                                                  \
  } while (0)

#define CHECK_KERNEL_OUTPUTS_NUM(actual_outputs_num, expect_outputs_num, kernel_name)                       \
  do {                                                                                                      \
    if ((actual_outputs_num) != (expect_outputs_num)) {                                                     \
      MS_LOG(EXCEPTION) << (kernel_name) << " should have " << (expect_outputs_num) << " outputs, but got " \
                        << (actual_outputs_num) << ".";                                                     \
    }                                                                                                       \
  } while (0)

#define CHECK_KERNEL_WORKSPACE_SIZE(actual_size, expect_size, kernel_name)                                           \
  do {                                                                                                               \
    if ((actual_size) != (expect_size)) {                                                                            \
      MS_LOG(EXCEPTION) << (kernel_name) << " requires " << (expect_size) << " workspace, but got " << (actual_size) \
                        << ".";                                                                                      \
    }                                                                                                                \
  } while (0)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_COMMON_UTILS_H_
