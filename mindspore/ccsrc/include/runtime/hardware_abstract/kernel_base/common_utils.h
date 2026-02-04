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
#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_COMMON_UTILS_H_

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
#include "include/utils/utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "runtime/hardware_abstract/visible.h"

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

RUNTIME_HARDWARE_EXPORT TypeId DtypeToTypeId(const std::string &dtypes);
RUNTIME_HARDWARE_EXPORT bool IsSameShape(const ShapeVector &shape_a, const ShapeVector &shape_b);
RUNTIME_HARDWARE_EXPORT bool CheckShapesSame(const ShapeArray &shape_array);

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

inline ShapeVector GetRealDims(const ShapeVector &input_shape) {
  ShapeVector dim_vector{};
  auto rank = SizeToLong(input_shape.size());
  for (int64_t i = 0; i < rank; i++) {
    dim_vector.emplace_back(i);
  }
  return dim_vector;
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

class RUNTIME_HARDWARE_EXPORT KernelAttr {
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

RUNTIME_HARDWARE_EXPORT size_t GetOutputNum(const AnfNodePtr &node);
RUNTIME_HARDWARE_EXPORT std::ostream &operator<<(std::ostream &os, KernelAttr kernel_attr);

RUNTIME_HARDWARE_EXPORT std::pair<bool, size_t> MatchKernelAttr(const KernelAttr &kernel_attr,
                                                                const std::vector<KernelAttr> &kernel_attr_list);
RUNTIME_HARDWARE_EXPORT std::pair<bool, size_t> MatchKernelAttrStrict(const KernelAttr &kernel_attr,
                                                                      const std::vector<KernelAttr> &kernel_attr_list);
RUNTIME_HARDWARE_EXPORT KernelAttr GetKernelAttrFromBuildInfo(const KernelBuildInfoPtr &build_info);
RUNTIME_HARDWARE_EXPORT KernelAttr GetKernelAttrFromTensors(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &outputs);
// Synchronize the output and input reference map between two kernel attrs.
RUNTIME_HARDWARE_EXPORT std::string FetchPrintInfoByKernelAttr(KernelAttr selected_kernel_attr);
RUNTIME_HARDWARE_EXPORT std::pair<std::vector<DataType>, std::vector<DataType>> GetInOutDataTypesFromKernelAttr(
  const KernelAttr &kernel_attr);
// Tuple --> Tuple.
RUNTIME_HARDWARE_EXPORT KernelObjectType TypeIdToKernelObjectType(const TypeId &type_id);
RUNTIME_HARDWARE_EXPORT std::vector<KernelObjectType> TypeIdToKernelObjectType(const std::vector<TypeId> &type_ids);
// Tuple --> TupleUnfold.
RUNTIME_HARDWARE_EXPORT KernelObjectType TypeIdToKernelObjectTypeForTupleUnfold(const TypeId &type_id);
RUNTIME_HARDWARE_EXPORT std::vector<KernelObjectType> TypeIdToKernelObjectTypeForTupleUnfold(
  const std::vector<TypeId> &type_ids);
RUNTIME_HARDWARE_EXPORT TypeId KernelObjectTypeToTypeId(const KernelObjectType &object_type);

RUNTIME_HARDWARE_EXPORT bool CheckAttrForAllSameInput(const size_t input_num,
                                                      const std::vector<mindspore::TypeId> &input_types,
                                                      const KernelAttr &cur_kernel_attr);

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

#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_COMMON_UTILS_H_
