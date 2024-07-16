/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include <sstream>
#include <limits>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <vector>
#include <utility>
#include <tuple>
#include "include/common/utils/utils.h"
#include "kernel/kernel.h"
#include "kernel/kash/kernel_pack.h"
#include "kernel/kernel_build_info.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace kernel {
constexpr auto kProcessorAiCore = "aicore";
constexpr auto kProcessorAiCpu = "aicpu";
constexpr auto kProcessorCuda = "cuda";
constexpr auto kProcessorCpu = "cpu";
constexpr auto kProcessorUnknown = "unknown";
constexpr unsigned int AUTODIFF_COMPILE_OVERTIME = 600;

// an enum to indicate a vector or matrix alignment direction.
// real_data: [1,2,3] left_align: [1,2,3,0] right_align:[0,1,2,3]
namespace MatrixDiag {
enum Alignment { RIGHT = 0, LEFT = 1 };
}  // namespace MatrixDiag

struct KernelMetaInfo {
  uintptr_t func_stub_;
  uint32_t block_dim_;
};
using KernelMetaPtr = std::shared_ptr<KernelMetaInfo>;

class MatrixInfo {
 public:
  explicit MatrixInfo(size_t max_index, const ShapeVector &matrix_shapes)
      : max_index_(max_index), shapes_(matrix_shapes) {
    current_indexes_.resize(shapes_.size(), 0);
  }
  ~MatrixInfo() = default;
  bool SetIndex(size_t start, size_t end) {
    // Check data from start to end whether valid.
    if (start < min_index || end > max_index_ || start >= end) {
      return false;
    }
    // Initial current indexes.
    int last_rank = SizeToInt(current_indexes_.size()) - 1;
    for (int i = last_rank; i >= 0; --i) {
      size_t position = IntToSize(i);
      current_indexes_[position] = start % LongToSize(shapes_.at(position));
      start = start / LongToSize(shapes_.at(position));
      if (start == 0) {
        break;
      }
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
    for (size_t i = last_rank; current_indexes_.at(i) >= LongToSize(shapes_.at(i)) && i > 0; --i) {
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
  ShapeVector shapes_;
  std::vector<size_t> current_indexes_;
};
using MatrixInfoPtr = std::shared_ptr<MatrixInfo>;
int CalDiagOffset(int diag_index, int max_diag_len, int inner_rows, int inner_cols,
                  const std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> &alignment);
TypeId DtypeToTypeId(const std::string &dtypes);
std::string Dtype2ShortType(const std::string &dtype);
BACKEND_EXPORT size_t GetDtypeNbyte(const std::string &dtype);
BACKEND_EXPORT bool IsSameShape(const ShapeVector &shape_a, const ShapeVector &shape_b);
BACKEND_EXPORT bool CheckShapesSame(const ShapeArray &shape_array);
BACKEND_EXPORT int ConvertReductionForAclnn(Reduction reduction);
std::string GetProcessorStr(const AnfNodePtr &anf_node);
template <typename T>
struct AsymmetricFunc {
  T operator()(const T &new_x, const int &old_length, const int &new_length) const {
    return new_length != 0 ? new_x * old_length / new_length : 0;
  }
};

template <template <typename, typename, typename...> typename M, typename T>
inline std::string Map2Str(const M<std::string, T> value) {
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

struct DataType {
  explicit DataType(const TypeId &dtype, const string &format = kOpFormat_DEFAULT,
                    const TypeId &object_type = kObjectTypeTensorType, bool is_optional = false)
      : dtype(dtype), format(format), object_type(object_type), is_optional(is_optional) {}
  TypeId dtype;
  std::string format;
  TypeId object_type;
  bool is_optional;
};

class BACKEND_EXPORT KernelAttr {
 public:
  KernelAttr() = default;
  ~KernelAttr() = default;

  KernelAttr &AddInputAttr(const TypeId &ms_type, const std::string &format = kOpFormat_DEFAULT);
  KernelAttr &AddOptionalInputAttr(const TypeId &ms_type, const std::string &format = kOpFormat_DEFAULT);
  KernelAttr &AddOutputAttr(const TypeId &ms_type, const std::string &format = kOpFormat_DEFAULT);
  KernelAttr &AddInputAttr(const TypeId &object_type, const TypeId &ms_type,
                           const std::string &formatt = kOpFormat_DEFAULT);
  KernelAttr &AddOptionalInputAttr(const TypeId &object_type, const TypeId &ms_type,
                                   const std::string &formatt = kOpFormat_DEFAULT);
  KernelAttr &AddOutputAttr(const TypeId &object_type, const TypeId &ms_type,
                            const std::string &formatt = kOpFormat_DEFAULT);
  KernelAttr &AddAllSameAttr(bool all_same, size_t all_same_input_num = 1, bool group_allsame = false);
  KernelAttr &AddSkipCheckAttr(bool skip_check);
  KernelAttr &AddRealTuple(const bool &is_real_tuple);
  KernelAttr &AddOutInRef(size_t output_index, size_t input_index);
  KernelAttr &AddAllOutInRef(bool all_out_in_ref);

  const DataType &GetInputAttr(const size_t index) const { return input_type_[index]; }
  const DataType &GetOutputAttr(const size_t index) const { return output_type_[index]; }
  bool GetAllSame() const { return all_same_; }
  bool GetSkipCheck() const { return skip_check_; }
  const bool &GetRealTuple() const { return is_real_tuple_; }
  bool GetGroupAllSame() const { return is_group_allsame_; }
  size_t GetAllSameInputNum() const { return all_same_input_num_; }
  size_t GetInputSize() const { return input_type_.size(); }
  size_t GetOutputSize() const { return output_type_.size(); }
  const OutputInputRefMap &GetOutInRefMap() const { return out_in_ref_map_; }
  bool GetAllOutInRef() const { return all_out_in_ref_; }

  void SetInputAttr(const size_t index, const TypeId &ms_type, const std::string &format);
  void SetOutputAttr(const size_t index, const TypeId &ms_type, const std::string &format);
  void SetInputAttrList(const std::vector<DataType> &addr_list);
  void SetOutputAttrList(const std::vector<DataType> &addr_list);

  const std::vector<DataType> &input_type() const { return input_type_; }
  const std::vector<DataType> &output_type() const { return output_type_; }

 private:
  std::vector<DataType> input_type_;
  std::vector<DataType> output_type_;
  bool all_same_{false};
  bool skip_check_{false};
  bool is_real_tuple_{false};
  bool is_group_allsame_{false};
  size_t all_same_input_num_{0};

  // The map between kernel's output and input ref relationship.
  OutputInputRefMap out_in_ref_map_;

  // The reference for all outputs and inputs of the same index.
  bool all_out_in_ref_{false};
};

BACKEND_EXPORT size_t GetOutputNum(const AnfNodePtr &node);
BACKEND_EXPORT std::ostream &operator<<(std::ostream &os, KernelAttr kernel_attr);

BACKEND_EXPORT std::pair<bool, size_t> MatchKernelAttr(const KernelAttr &kernel_attr,
                                                       const std::vector<KernelAttr> &kernel_attr_list);
BACKEND_EXPORT std::pair<bool, size_t> MatchKernelAttrStrict(const KernelAttr &kernel_attr,
                                                             const std::vector<KernelAttr> &kernel_attr_list);
BACKEND_EXPORT KernelAttr GetKernelAttrFromBuildInfo(const KernelBuildInfoPtr &build_info);
BACKEND_EXPORT KernelAttr GetKernelAttrFromNode(const AnfNodePtr &kernel_node);
BACKEND_EXPORT bool IsFoldKernelBuildInfo(const KernelBuildInfoPtr &kernel_build_info);
BACKEND_EXPORT KernelAttr GetKernelAttrFromTensors(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs);
void SetCpuRefMapToKernelInfo(const CNodePtr &apply_kernel, const std::vector<KernelAttr> &apply_kernel_attrs);
// Synchronize the output and input reference map between two kernel attrs.
void SyncOutInRef(const KernelAttr &from_kernel_attr, KernelAttr *to_kernel_attr);
BACKEND_EXPORT std::string FetchPrintInfoByKernelAttr(KernelAttr selected_kernel_attr);
std::vector<TypeId> GetInputObjectTypeListFromKernelAttr(const KernelAttr &kernel_attr);
std::vector<TypeId> GetOutputObjectTypeListFromKernelAttr(const KernelAttr &kernel_attr);
// The related interfaces of kernel object type.
BACKEND_EXPORT void SetKernelObjectTypeBuildInfo(const AnfNodePtr &kernel_node,
                                                 const std::vector<KernelObjectType> &input_kernel_object_types,
                                                 const std::vector<KernelObjectType> &output_kernel_object_types);
BACKEND_EXPORT void SetKernelObjectTypeWithSelectedAttr(const CNodePtr &kernel_node,
                                                        const kernel::KernelAttr &selected_kernel_attr);
BACKEND_EXPORT bool SelectKernelByObjectType(const CNodePtr &kernel_node,
                                             const std::vector<KernelAttr> &registered_kernel_attrs,
                                             std::vector<KernelAttr> *selected_kernel_attrs);
// Tuple --> Tuple.
BACKEND_EXPORT KernelObjectType TypeIdToKernelObjectType(const TypeId &type_id);
BACKEND_EXPORT std::vector<KernelObjectType> TypeIdToKernelObjectType(const std::vector<TypeId> &type_ids);
// Tuple --> TupleUnfold.
BACKEND_EXPORT KernelObjectType TypeIdToKernelObjectTypeForTupleUnfold(const TypeId &type_id);
BACKEND_EXPORT std::vector<KernelObjectType> TypeIdToKernelObjectTypeForTupleUnfold(
  const std::vector<TypeId> &type_ids);
BACKEND_EXPORT TypeId KernelObjectTypeToTypeId(const KernelObjectType &object_type);

BACKEND_EXPORT bool CheckAttrForAllSameInput(const size_t input_num, const std::vector<mindspore::TypeId> &input_types,
                                             const KernelAttr &cur_kernel_attr);

BACKEND_EXPORT void SetKernelObjectTypeBuildInfo(
  const AnfNodePtr &kernel_node, const std::vector<KernelObjectType> &input_kernel_object_types,
  const std::vector<KernelObjectType> &output_kernel_object_types,
  const std::vector<KernelObjectType> &output_elements_kernel_object_types);

template <typename Derived>
class MatchKernelHelper {
 public:
  MatchKernelHelper() = default;
  virtual ~MatchKernelHelper() = default;

  using KernelRunFunc = std::function<bool(Derived *, const std::vector<KernelTensor *> &,
                                           const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;
  virtual const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const = 0;

 protected:
  std::vector<KernelAttr> OpSupport() const {
    auto &func_list = static_cast<const Derived *>(this)->GetFuncList();
    std::vector<KernelAttr> support_list;
    (void)std::transform(func_list.begin(), func_list.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, KernelRunFunc> &pair) { return pair.first; });
    return support_list;
  }

  bool MatchKernelFunc(const std::string &kernel_name, const std::vector<KernelTensor *> &inputs,
                       const std::vector<KernelTensor *> &outputs) {
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto &func_list = static_cast<Derived *>(this)->GetFuncList();
    auto [is_match, index] = MatchKernelAttr(kernel_attr, OpSupport());
    if (!is_match) {
      MS_LOG(ERROR) << "The kernel '" << kernel_name << "' does not support this kernel data type: " << kernel_attr;
      return false;
    }
    kernel_func_ = func_list[index].second;
    return true;
  }

  KernelRunFunc kernel_func_;
};

namespace math {
BACKEND_EXPORT void SinCosf(float x, float *sinv, float *cosv);
}

inline void GetRawAddress(const std::vector<AddressPtr> &addrs, std::vector<void *> *raw_addrs) {
  (void)std::transform(std::begin(addrs), std::end(addrs), std::back_inserter(*raw_addrs),
                       [](const AddressPtr &address) -> void * {
                         MS_EXCEPTION_IF_NULL(address);
                         return address->addr;
                       });
}

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
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_COMMON_UTILS_H_
