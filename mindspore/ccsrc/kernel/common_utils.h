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
#include <limits>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <utility>
#include <tuple>
#include <nlohmann/json.hpp>
#include "include/common/utils/utils.h"
#include "kernel/kernel.h"
#include "kernel/oplib/opinfo.h"
#include "kernel/kernel_build_info.h"
#include "ops/base_operator.h"
#include "ops/strided_slice.h"

namespace mindspore {
namespace kernel {
constexpr auto kAkgKernelMeta = "akg_kernel_meta/";
constexpr auto kProcessorAiCore = "aicore";
constexpr auto kProcessorAiCpu = "aicpu";
constexpr auto kProcessorCuda = "cuda";
constexpr auto kProcessorCpu = "cpu";
constexpr auto kProcessorUnknown = "unknown";
constexpr auto kJsonSuffix = ".json";
constexpr auto kInfoSuffix = ".info";
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

class BACKEND_EXPORT KernelMeta {
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
std::set<int64_t> GetShapeSetFromResizeMap(const CNodePtr &node);
BACKEND_EXPORT std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> GetAlignments(const std::string &alignment);
int CalDiagOffset(int diag_index, int max_diag_len, int inner_rows, int inner_cols,
                  const std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> &alignment);
BACKEND_EXPORT std::string GetCompilerCachePath();
bool CheckCache(const std::string &kernel_name);
KernelPackPtr SearchCache(const std::string &kernel_name, const std::string &processor);
KernelPackPtr InsertCache(const std::string &kernel_name, const std::string &processor);
TypeId DtypeToTypeId(const std::string &dtypes);
std::string Dtype2ShortType(const std::string &dtype);
BACKEND_EXPORT size_t GetDtypeNbyte(const std::string &dtype);
BACKEND_EXPORT bool GetShapeSize(const ShapeVector &shape, const TypePtr &type_ptr, int64_t *size_i);
BACKEND_EXPORT bool ParseMetadata(const CNodePtr &kernel_node, const std::shared_ptr<const OpInfo> &op_info_ptr,
                                  Processor processor,
                                  std::vector<std::shared_ptr<KernelBuildInfo>> *const kernel_info_list);
BACKEND_EXPORT void SaveJsonInfo(const std::string &json_name, const std::string &info, const std::string &base_path);
std::string GetProcessor(const AnfNodePtr &anf_node);
Processor GetProcessor(const string &processor);
BACKEND_EXPORT bool IsSameShape(const ShapeVector &shape_a, const ShapeVector &shape_b);
BACKEND_EXPORT bool CheckShapesSame(const ShapeArray &shape_array);
BACKEND_EXPORT std::vector<std::pair<AnfNodePtr, size_t>> GetOutputIndex(const std::vector<AnfNodePtr> &node_list,
                                                                         const std::vector<AnfNodePtr> &input_list,
                                                                         const std::vector<AnfNodePtr> &output_list);
BACKEND_EXPORT void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list);
BACKEND_EXPORT void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list,
                                        std::vector<AnfNodePtr> *input_list, std::vector<AnfNodePtr> *output_list);
void GetFuncGraphOutputNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *output_list);
void GetGraphRealOutput(const FuncGraphPtr &func_graph, std::vector<std::pair<AnfNodePtr, size_t>> *node_list);
BACKEND_EXPORT bool IsWeightBoundary(const AnfNodePtr &node);
BACKEND_EXPORT std::vector<int64_t> GetReduceAttrAxis(const CNodePtr &cnode);
std::string GetProcessorStr(const AnfNodePtr &anf_node);
Processor GetProcessorFromContext();
std::string GetStrProcessorFromContext();
BACKEND_EXPORT float Scaling(size_t in_size, size_t out_size, bool align_corners);
inline float Scaler(const size_t x, const float scale, bool half_pixel_centers) {
  if (half_pixel_centers) {
    /**
     * function with a std::floor(), so instead of subtracting the 0.5 as we
     * do in HalfPixelScale, we leave it as is, as the std::floor does the
     * correct thing.
     * */
    return (static_cast<float>(x) + 0.5f) * scale;
  } else {
    /**
     * Older incorrect scaling method that causes all resizes to have a slight
     * translation leading to inconsistent results. For example, a flip then a
     * resize gives different results then a resize then a flip.
     * */
    return static_cast<float>(x) * scale;
  }
}
float ScaleGrid(const int x, const float scale);
BACKEND_EXPORT std::vector<bool> Dec2Bin(const int64_t &mask);
BACKEND_EXPORT void FillEmptyDims(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin,
                                  std::vector<int64_t> *end, std::vector<int64_t> *stride, ShapeVector *input_shape);
BACKEND_EXPORT void ParseStrideSliceMasks(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin,
                                          std::vector<int64_t> *end, std::vector<int64_t> *stride,
                                          const ShapeVector &input_shape);
struct CachedInterpolation {
  size_t lower;
  size_t upper;
  float lerp;
};

template <typename T>
struct AlignCornersFunc {
  T operator()(const T &new_x, const int &old_length, const int &new_length) const {
    return new_length != 1 ? new_x * (old_length - 1) / (new_length - 1) : 0;
  }
};

template <typename T>
struct AsymmetricFunc {
  T operator()(const T &new_x, const int &old_length, const int &new_length) const {
    return new_length != 0 ? new_x * old_length / new_length : 0;
  }
};

template <typename T>
struct HalfPixelFunc {
  T operator()(const T &new_x, const int &old_length, const int &new_length) const {
    constexpr auto half_pixel = 0.5;
    return new_length > 1 ? (new_x + half_pixel) * old_length / new_length - half_pixel : 0;
  }
};

void ComputeInterpolationWeights(const size_t out_size, const size_t in_size, const float scale,
                                 CachedInterpolation *interpolation, bool half_pixel_centers);

template <typename T>
inline T ComputeScales(const double &scale, const size_t &input_size, const size_t &output_size) {
  if (scale > 0.) {
    return static_cast<T>(1.0 / scale);
  } else if (output_size > 0) {
    return (static_cast<T>(input_size) / output_size);
  }
  return 0;
}

inline size_t NearestNeighborSourceIndex(const float &scale, const size_t &dst_index, const size_t &input_size) {
  size_t src_index = std::min(static_cast<size_t>(floorf(dst_index * scale)), input_size - 1);
  return src_index;
}

inline size_t NearestIndex(const size_t &output_index, const size_t &input_size, const size_t &output_size,
                           const double &scales) {
  constexpr size_t kNumberTwo = 2;
  if (output_size == input_size) {
    // scale_factor = 1
    return output_index;
  } else if (output_size == kNumberTwo * input_size) {
    // scale_factor = 2, shift input index
    return output_index >> 1;
  } else {
    float scale = ComputeScales<float>(scales, input_size, output_size);
    return NearestNeighborSourceIndex(scale, output_index, input_size);
  }
}

template <typename T>
inline T AreaPixelComputeScale(int64_t input_size, int64_t output_size, bool align_corners, double scale) {
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<T>(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<T>(0);
    }
  } else {
    return ComputeScales<T>(scale, input_size, output_size);
  }
}

template <typename T>
inline T AreaPixelComputeSourceIndex(T scale, int64_t dst_index, bool align_corners) {
  if (align_corners) {
    return scale * static_cast<T>(dst_index);
  } else {
    constexpr T zero = 0.;
    T src_idx = scale * (dst_index + 0.5) - 0.5;
    return src_idx < zero ? zero : src_idx;
  }
}

template <typename T>
inline void ComputeSourceIndexAndLambda(int64_t *const input_index0, int64_t *const input_index1, T *const lambda0,
                                        T *const lambda1, T ratio, int64_t output_index, int64_t input_size,
                                        int64_t output_size, bool align_corners) {
  if (output_size == input_size) {
    // scale_factor = 1
    *input_index0 = output_index;
    *input_index1 = output_index;
    *lambda0 = static_cast<T>(1);
    *lambda1 = static_cast<T>(0);
  } else {
    const T real_input_index = AreaPixelComputeSourceIndex<T>(ratio, output_index, align_corners);
    *input_index0 = static_cast<int64_t>(real_input_index);
    int64_t offset = (*input_index0 < input_size - 1) ? 1 : 0;
    *input_index1 = *input_index0 + offset;
    *lambda1 = real_input_index - static_cast<T>(*input_index0);
    constexpr T one = 1.0;
    *lambda0 = one - *lambda1;
  }
}

template <typename T>
inline std::string Vector2Str(const std::vector<T> &inputs) {
  if (!inputs.empty()) {
    std::ostringstream oss;
    oss << "(";
    (void)std::copy(inputs.begin(), inputs.end() - 1, std::ostream_iterator<T>(oss, ", "));
    oss << inputs.back();
    oss << ")";
    return oss.str();
  }
  return "";
}

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

template <typename T>
inline T ComputeLerp(T top_left, T top_right, T bottom_left, T bottom_right, T x_lerp, T y_lerp) {
  T top = top_left + (top_right - top_left) * x_lerp;
  T bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

BACKEND_EXPORT void CheckSliceValid(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                                    const std::vector<int64_t> &step, const std::vector<int64_t> &input_shape);
BACKEND_EXPORT size_t CalOffset(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                                const std::vector<int64_t> &dim_offset);
BACKEND_EXPORT std::vector<int64_t> CalDimOffset(const std::vector<int64_t> &input_shape);
BACKEND_EXPORT size_t GetCopySize(const std::vector<int64_t> &dim_offset, const std::vector<int64_t> &start,
                                  const std::vector<int64_t> &stop);
BACKEND_EXPORT size_t UnitSizeInBytes(const mindspore::TypeId &t);

struct DataType {
  explicit DataType(const TypeId &dtype, const string &format = kOpFormat_DEFAULT,
                    const TypeId &object_type = kObjectTypeTensorType)
      : dtype(dtype), format(format), object_type(object_type) {}
  TypeId dtype;
  std::string format;
  TypeId object_type;
};

class BACKEND_EXPORT KernelAttr {
 public:
  KernelAttr() = default;
  ~KernelAttr() = default;

  KernelAttr &AddInputAttr(const TypeId &ms_type, const std::string &format = kOpFormat_DEFAULT);
  KernelAttr &AddOutputAttr(const TypeId &ms_type, const std::string &format = kOpFormat_DEFAULT);
  KernelAttr &AddInputAttr(const TypeId &object_type, const TypeId &ms_type,
                           const std::string &formatt = kOpFormat_DEFAULT);
  KernelAttr &AddOutputAttr(const TypeId &object_type, const TypeId &ms_type,
                            const std::string &formatt = kOpFormat_DEFAULT);
  KernelAttr &AddAllSameAttr(const bool &all_same);
  KernelAttr &AddSkipCheckAttr(const bool &skip_check);
  KernelAttr &AddOutInRef(size_t output_index, size_t input_index);
  KernelAttr &AddAllOutInRef(const bool &all_out_in_ref);

  const DataType &GetInputAttr(const size_t index) const { return input_type_[index]; }
  const DataType &GetOutputAttr(const size_t index) const { return output_type_[index]; }
  const bool &GetAllSame() const { return all_same_; }
  const bool &GetSkipCheck() const { return skip_check_; }

  size_t GetInputSize() const { return input_type_.size(); }
  size_t GetOutputSize() const { return output_type_.size(); }
  const OutputInputRefMap &GetOutInRefMap() const { return out_in_ref_map_; }
  const bool &GetAllOutInRef() const { return all_out_in_ref_; }

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

  // The map between kernel's output and input ref relationship.
  OutputInputRefMap out_in_ref_map_;

  // The reference for all outputs and inputs of the same index.
  bool all_out_in_ref_{false};
};
BACKEND_EXPORT std::ostream &operator<<(std::ostream &os, KernelAttr kernel_attr);

BACKEND_EXPORT std::pair<bool, size_t> MatchKernelAttr(const KernelAttr &kernel_attr,
                                                       const std::vector<KernelAttr> &kernel_attr_list);
BACKEND_EXPORT std::pair<bool, size_t> MatchKernelAttrStrict(const KernelAttr &kernel_attr,
                                                             const std::vector<KernelAttr> &kernel_attr_list);
BACKEND_EXPORT KernelAttr GetKernelAttrFromBuildInfo(const KernelBuildInfoPtr &build_info);
BACKEND_EXPORT KernelAttr GetKernelAttrFromNode(const AnfNodePtr &kernel_node);

struct KernelArgs {
  BaseOperatorPtr op;
  std::vector<KernelTensorPtr> inputs;
  std::vector<KernelTensorPtr> outputs;
  std::map<uint32_t, tensor::TensorPtr> depend_tensor_map;  // dynamic shape kernel may need this map
  // cppcheck-suppress unusedStructMember
  constexpr static char key[] = "KernelArgs";
};
BACKEND_EXPORT KernelArgs AbstractArgsFromCNode(const CNodePtr &cnode, bool is_without_operator = false);

BACKEND_EXPORT KernelAttr GetKernelAttrFromTensors(const std::vector<KernelTensorPtr> &inputs,
                                                   const std::vector<KernelTensorPtr> &outputs);
void SetCpuRefMapToKernelInfo(const CNodePtr &apply_kernel, const std::vector<KernelAttr> &apply_kernel_attrs);
Format GetFormatFromStrToEnum(const std::string &format_str);
BACKEND_EXPORT std::string GetFormatFromEnumToStr(Format format);
BACKEND_EXPORT void UpdateNodeShape(const CNodePtr &cnode);
// Synchronize the output and input reference map between two kernel attrs.
void SyncOutInRef(const KernelAttr &from_kernel_attr, KernelAttr *to_kernel_attr);
BACKEND_EXPORT std::shared_ptr<KernelArgs> GetArgsFromCNode(const CNodePtr &cnode);
BACKEND_EXPORT void SetArgsToCNode(const CNodePtr &cnode, const KernelArgs &args);
BACKEND_EXPORT void SetInputsByDependMap(const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map,
                                         std::vector<KernelTensorPtr> *inputs, bool is_stored_in_device = false);
BACKEND_EXPORT void SetInputsByConstInputs(const CNodePtr &node,
                                           std::map<uint32_t, tensor::TensorPtr> *inputs_tensor_map);
BACKEND_EXPORT bool IfNeedSkipResize(const CNodePtr &node);

inline std::map<uint32_t, tensor::TensorPtr> GetKernelDepends(const CNodePtr &cnode) {
  auto args = GetArgsFromCNode(cnode);
  if (args) {
    return args->depend_tensor_map;
  }
  return std::map<uint32_t, tensor::TensorPtr>();
}

BACKEND_EXPORT std::string FetchPrintInfoByKernelAttr(KernelAttr selected_kernel_attr);
// The related interfaces of kernel object type.
BACKEND_EXPORT void SetKernelObjectTypeBuildInfo(const AnfNodePtr &kernel_node,
                                                 const std::vector<KernelObjectType> &input_kernel_object_types,
                                                 const std::vector<KernelObjectType> &output_kernel_object_types);
BACKEND_EXPORT void SetKernelObjectTypeWithSelectedAttr(const CNodePtr &kernel_node,
                                                        const kernel::KernelAttr &selected_kernel_attr);
BACKEND_EXPORT bool SelectKernelByObjectType(const CNodePtr &kernel_node,
                                             const std::vector<KernelAttr> &ori_kernel_attrs,
                                             std::vector<KernelAttr> *selected_kernel_attrs, bool strict);
// Tuple --> Tuple.
BACKEND_EXPORT KernelObjectType TypeIdToKernelObjectType(const TypeId &type_id);
BACKEND_EXPORT std::vector<KernelObjectType> TypeIdToKernelObjectType(const std::vector<TypeId> &type_ids);
// Tuple --> TupleUnfold.
BACKEND_EXPORT KernelObjectType TypeIdToKernelObjectTypeForTupleUnfold(const TypeId &type_id);
BACKEND_EXPORT std::vector<KernelObjectType> TypeIdToKernelObjectTypeForTupleUnfold(
  const std::vector<TypeId> &type_ids);
BACKEND_EXPORT TypeId KernelObjectTypeToTypeId(const KernelObjectType &object_type);
KernelObjectType StringToKernelObjectType(const std::string &object_type);
BACKEND_EXPORT void UnfoldKernelBuildInfo(const CNodePtr &kernel_node);
BACKEND_EXPORT void SetDynamicInputSizeAttr(const CNodePtr &cnode);
BACKEND_EXPORT bool IsDynamicParamKernel(const std::string &op_name);

template <typename Derived>
class MatchKernelHelper {
 public:
  MatchKernelHelper() = default;
  virtual ~MatchKernelHelper() = default;

  using KernelRunFunc = std::function<bool(Derived *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &)>;
  virtual const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const = 0;

 protected:
  std::vector<KernelAttr> OpSupport() const {
    auto &func_list = static_cast<const Derived *>(this)->GetFuncList();
    std::vector<KernelAttr> support_list;
    (void)std::transform(func_list.begin(), func_list.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, KernelRunFunc> &pair) { return pair.first; });
    return support_list;
  }
  bool MatchKernelFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                       const std::vector<KernelTensorPtr> &outputs) {
    auto kernel_name = base_operator->name();
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

namespace broadcast_utils {
BACKEND_EXPORT bool AlignedBroadCastShape(size_t align_rank, std::vector<size_t> *broadcast, std::vector<size_t> *lhs,
                                          std::vector<size_t> *rhs);
}  // namespace broadcast_utils

namespace math {
BACKEND_EXPORT void SinCosf(float x, float *sinv, float *cosv);
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
