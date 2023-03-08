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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CPU_KERNEL_H_

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <map>
#include <set>

#include "kernel/kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "ir/anf.h"
#include "actor/actormgr.h"
#include "include/common/thread_pool.h"
#include "include/backend/visible.h"

using mindspore::kernel::Address;
using mindspore::kernel::AddressPtr;
using CTask = std::function<void(size_t, size_t)>;
namespace mindspore {
namespace kernel {
constexpr char KERNEL_SIZE[] = "kernel_size";
constexpr char VALIDATE_INDICES[] = "validate_indices";
constexpr char STRIDE[] = "stride";
constexpr char STRIDES[] = "strides";
constexpr char DILATION[] = "dilation";
constexpr char DILATIONS[] = "dilations";
constexpr char FORMAT[] = "format";
constexpr char PAD[] = "pad";
constexpr char PAD_LIST[] = "pad_list";
constexpr char PAD_MODE[] = "pad_mode";
constexpr char PAD_MODE_LOWER_SAME[] = "same";
constexpr char PAD_MODE_LOWER_VALID[] = "valid";
constexpr char PAD_MODE_UPPER_SAME[] = "SAME";
constexpr char PAD_MODE_UPPER_VALID[] = "VALID";
constexpr char COUNT_INCLUDE_PAD[] = "count_include_pad";
constexpr char CEIL_MODE[] = "ceil_mode";
constexpr char DIVISOR_OVERRIDE[] = "divisor_override";
constexpr char TRANSPOSE_A[] = "transpose_a";
constexpr char TRANSPOSE_B[] = "transpose_b";
constexpr char IS_GRAD[] = "is_grad";
constexpr char TRANSPOSE_NO = 'N';
constexpr char TRANSPOSE_YES = 'T';
constexpr char AXIS[] = "axis";
constexpr char DIM[] = "dim";
constexpr char NUM[] = "num";
constexpr char BEGIN[] = "begin";
constexpr char END[] = "end";
constexpr char SIZE[] = "size";
constexpr char USE_NESTEROV[] = "use_nesterov";
constexpr char GROUP[] = "group";
constexpr char START[] = "start";
constexpr char LIMIT[] = "limit";
constexpr char DELTA[] = "delta";
constexpr char SORTED[] = "sorted";
constexpr char ADJ_ST[] = "adjoint_st";
constexpr char ADJ_dT[] = "adjoint_dt";
constexpr char REDUCTION[] = "reduction";
constexpr char NONE[] = "none";
constexpr char SUM[] = "sum";
constexpr char MEAN[] = "mean";
constexpr char BETA[] = "beta";
constexpr char EXCLUSIVE[] = "exclusive";
constexpr char REVERSE[] = "reverse";
constexpr char PCR[] = "preprocess_collapse_repeated";
constexpr char CTR[] = "ctc_merge_repeated";
constexpr char ILOTI[] = "ignore_longer_outputs_than_inputs";
constexpr char MOMENTUM[] = "momentum";
constexpr char RHO[] = "rho";
constexpr char EPSILON[] = "epsilon";
constexpr char ALIGN_CORNERS[] = "align_corners";
constexpr char PERIODS[] = "periods";
constexpr char WINDOW[] = "window";
constexpr char MIN_PERIODS[] = "min_periods";
constexpr char CENTER[] = "center";
constexpr char METHOD[] = "method";
constexpr char CLOSED[] = "closed";
constexpr char NA_OPTION[] = "na_option";
constexpr char ASCENDING[] = "ascending";
constexpr char PCT[] = "pct";
constexpr char LOWER[] = "lower";
constexpr char CLEAN[] = "clean";
constexpr char TRANS[] = "trans";
constexpr char MODE[] = "mode";
constexpr char UNIT_DIAGONAL[] = "unit_diagonal";
constexpr char C_EIEH_VECTOR[] = "compute_eigenvectors";
constexpr char COMPUTE_V[] = "compute_v";
constexpr char ADJOINT[] = "adjoint";
constexpr char ALIGNMENT[] = "alignment";
constexpr char NCHW[] = "NCHW";
constexpr char NCDHW[] = "NCDHW";
constexpr char USE_LOCKING[] = "use_locking";
constexpr char OP[] = "op";
constexpr char SET_OPERATION[] = "set_operation";

constexpr size_t NC_LEN = 2;
constexpr size_t SHAPE_4D = 4;
constexpr size_t SHAPE_5D = 5;
constexpr size_t N_INDEX = 0;
constexpr size_t C_INDEX = 1;
constexpr size_t D_INDEX = 2;
constexpr size_t H_INDEX = 3;
constexpr size_t W_INDEX = 4;

struct ParallelSearchInfo {
  double min_cost_time{DBL_MAX};
  double tmp_sum_cost_time{0.f};
  float best_block_size{0.f};
  size_t best_pow{0};
  size_t search_count{0};
  bool kernel_thread_num_set{false};
  size_t max_pow{6};
};

class BACKEND_EXPORT NativeCpuKernelMod : public CpuKernelMod {
 public:
  NativeCpuKernelMod() = default;
  ~NativeCpuKernelMod() override = default;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void * /*stream_ptr*/) override {
    return Launch(inputs, workspace, outputs);
  }
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs);
  virtual bool Launch(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
                      const std::vector<AddressPtr> &workspace) {
    return false;
  }

  // Must be called before Init.
  void SetThreadPool(ThreadPool *pool) { pool_ = pool; }

  static std::vector<KernelAttr> GetCpuSupportedList(const std::string &kernel_name) {
    auto temp_mod = kernel::Factory<NativeCpuKernelMod>::Instance().Create(kernel_name);
    if (temp_mod == nullptr) {
      MS_LOG(INFO) << "Not register CPU kernel of operator: " << kernel_name;
      return std::vector<KernelAttr>{};
    }
    return temp_mod->GetAllSupportedList(kernel_name);
  }

  std::vector<KernelAttr> GetOpSupport() { return {}; }

  enum KernelModType GetKernelModType() const override { return KernelModType::NativeCpuKernelMod; }

  ParallelSearchInfo parallel_search_info_;

 protected:
  ThreadPool *pool_{nullptr};

 private:
  std::vector<KernelAttr> GetAllSupportedList(const std::string &kernel_name);
  std::vector<KernelAttr> GetSupportFromOpLib(const std::string &kernel_name) const;
  inline static mindspore::HashMap<std::string, std::vector<KernelAttr>> support_map_;
};

class BACKEND_EXPORT DeprecatedNativeCpuKernelMod : public NativeCpuKernelMod {
 public:
  DeprecatedNativeCpuKernelMod() = default;
  ~DeprecatedNativeCpuKernelMod() override = default;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;
  virtual void Init(const CNodePtr &kernel_node);
  virtual void InitKernel(const CNodePtr &kernel_node) = 0;

  void SetCNodePtr(const CNodePtr &kernel_node) { cnode_ptr_ = kernel_node; }
  const CNodeWeakPtr &GetCNodePtr() { return cnode_ptr_; }

  void SetCpuRefMapToKernelInfo(const CNodePtr &apply_kernel);
  enum KernelModType GetKernelModType() const override { return KernelModType::DeprecatedNativeCpuKernelMod; }

 protected:
  virtual void InitInputOutputSize(const CNodePtr &kernel_node);
  CNodeWeakPtr cnode_ptr_;

  template <typename T>
  inline T *GetDeviceAddress(const std::vector<AddressPtr> &addr_list, size_t index) {
    if (index >= addr_list.size()) {
      MS_LOG(EXCEPTION) << "Address index must be in range(" << addr_list.size() << "), but got " << index << ".";
    }

    if ((addr_list[index] == nullptr) || (addr_list[index]->addr == nullptr) || (addr_list[index]->size == 0)) {
      MS_LOG(EXCEPTION) << "The device address is empty. Address index is " << index
                        << ", and the length of 'addr_list' is " << addr_list.size();
    }

    return reinterpret_cast<T *>(addr_list[index]->addr);
  }

 private:
  std::vector<TypeId> GetInputDtypes(const CNodePtr &kernel_node) const;
  std::vector<TypeId> GetOutputDtypes(const CNodePtr &kernel_node) const;
};

class DeprecatedCpuKernelFunc {
 public:
  DeprecatedCpuKernelFunc() = default;
  virtual ~DeprecatedCpuKernelFunc() = default;
  virtual bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                       const std::vector<AddressPtr> &outputs) = 0;
  virtual void InitFunc(const CNodePtr &kernel_node) {}
  virtual void InitInputOutputSize(const CNodePtr &kernel_node, std::vector<size_t> *input_size_list,
                                   std::vector<size_t> *output_size_list, std::vector<size_t> *workspace_size_list) {}
  ParallelSearchInfo parallel_search_info_;
};

class CpuKernelFunc {
 public:
  CpuKernelFunc() = default;
  virtual ~CpuKernelFunc() = default;
  virtual void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                        const std::vector<KernelTensorPtr> &outputs) {}
  virtual int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) {
    return KRET_OK;
  }
  virtual bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                       const std::vector<AddressPtr> &outputs) = 0;
  ParallelSearchInfo parallel_search_info_;
};

class CPUKernelUtils {
 public:
  static void ExpandDimsTo4(ShapeVector *shape);
  static size_t CalcOffset(const ShapeVector &shape, size_t dim0, size_t dim1, size_t dim2, size_t dim3);
  static size_t GetElementNumOnAxis(const ShapeVector &shape, int axis);
  static void GetElementNumEveryDim(const ShapeVector &shape, std::vector<size_t> *element_num);
  static void ParallelFor(const CTask &task, size_t count, float block_size = 128.0);
  static ShapeVector FlatShapeByAxis(const ShapeVector &shape, int axis);
  static ShapeVector GetBroadcastShape(const std::vector<int64_t> &x, const std::vector<int64_t> &y);
  static void ParallelForAutoSearch(const CTask &task, size_t count, ParallelSearchInfo *parallel_search_info);
  template <typename T>
  inline static T CalcElementNum(const std::vector<T> &shape) {
    T total = std::accumulate(shape.begin(), shape.end(), T(1), std::multiplies<T>());
    return total;
  }
  template <typename T>
  inline static std::vector<int64_t> CalcSegmentIds(const T *segment_ids_data_addr, const size_t segment_ids_num) {
    std::vector<int64_t> segments;
    int64_t seg_tmp = 1;
    for (size_t i = 0; i < segment_ids_num - 1; ++i) {
      if (segment_ids_data_addr[i] == segment_ids_data_addr[i + 1]) {
        seg_tmp++;
      } else {
        segments.push_back(seg_tmp);
        seg_tmp = 1;
      }
      const size_t last_loc = 2;
      if (i == segment_ids_num - last_loc) {
        segments.push_back(seg_tmp);
      }
    }
    if (segment_ids_num == 1) {
      segments.push_back(seg_tmp);
    }
    return segments;
  }
};

class BroadcastIterator {
 public:
  BroadcastIterator(ShapeVector input_shape_a, ShapeVector input_shape_b, ShapeVector output_shape);
  virtual ~BroadcastIterator() = default;
  inline size_t GetInputPosA() const { return input_pos_[0]; }
  inline size_t GetInputPosB() const { return input_pos_[1]; }
  void SetPos(size_t pos);
  void GenNextPos();

 private:
  void BroadcastShape();
  void InitStrides();

  ShapeVector coordinates_;
  ShapeVector input_shape_a_;
  ShapeVector input_shape_b_;
  ShapeVector output_shape_;
  ShapeVector input_strides_a_;
  ShapeVector input_strides_b_;
  ShapeVector input_back_strides_a_;
  ShapeVector input_back_strides_b_;
  std::array<size_t, 2> input_pos_{0};
  int output_dimension_{0};
};

void GetBroadCastIndex(const std::vector<size_t> &unaligned_input_shape, const std::vector<size_t> &output_shape,
                       std::vector<size_t> *index_list);

// Broadcast for multi_inputs and single output
class MultipleBroadcastIterator {
 public:
  using shape_info = ShapeVector;
  MultipleBroadcastIterator(std::vector<shape_info> multi_inputs, shape_info output_shape);
  virtual ~MultipleBroadcastIterator() = default;
  inline size_t GetInputPos(size_t index) const { return LongToSize(input_pos_[index]); }
  void SetPos(size_t pos);
  void GenNextPos();

 private:
  void BroadcastShape();
  void InitStrides();

  shape_info coordinates_;
  std::vector<shape_info> multi_inputs_;
  shape_info output_shape_;
  std::vector<shape_info> multi_inputs_strides_;
  std::vector<shape_info> multi_inputs_back_strides_;
  shape_info input_pos_;
  int output_dimension_{0};
};

class TransposeIterator {
 public:
  TransposeIterator(ShapeVector output_shape, std::vector<size_t> axes, const ShapeVector &input_shape);
  virtual ~TransposeIterator() = default;
  inline size_t GetPos() const { return pos_; }
  void SetPos(size_t pos);
  void GenNextPos();

 private:
  int dimension_{0};
  ShapeVector coordinates_;
  ShapeVector shape_;
  ShapeVector strides_;
  ShapeVector back_strides_;
  std::vector<size_t> axes_;
  size_t pos_{0};
};

ActorThreadPool *GetActorMgrInnerThreadPool();
void ParallelLaunch(const CTask &task, size_t count, float block_size = 128.0, Content content = nullptr,
                    ThreadPool *pool = nullptr);
void ParallelLaunch(const std::vector<common::Task> &tasks, Content content = nullptr, ThreadPool *pool = nullptr);
void ParallelLaunchAutoSearch(const CTask &task, size_t count, Content content,
                              ParallelSearchInfo *parallel_search_info, ThreadPool *pool = nullptr);

// Deal with pytorch style axis iteration, to iterate every value on specific axis
class AxisIterator {
 public:
  AxisIterator() = default;
  virtual ~AxisIterator() = default;
  void Init(const ShapeVector &input_shape, size_t axis);
  // Iterate index through outer_size_ * inner_size_, combine inner iteration and outer iteration
  // into one single iteration to fit ParallelLaunchAutoSearch
  // Possible usage:
  // for (i = 0; i < outer_size_ * inner_size_; i ++) {
  //    axisIterator.SetOffset(i);
  //    // Do computation
  // }
  inline void SetOffset(size_t index) {
    size_t outer_index = index / inner_size_;
    size_t inner_index = index % inner_size_;
    axis_offset_ = outer_index * axis_size_ * inner_size_ + inner_index;
  }

  inline void SetOffset(size_t outer_index, size_t inner_index) {
    axis_offset_ = outer_index * axis_size_ * inner_size_ + inner_index;
  }
  inline size_t GetPos(size_t i) const { return axis_offset_ + i * inner_size_; }
  inline size_t RevertPos(size_t i) const { return (i - axis_offset_) / inner_size_; }

  inline size_t OuterSize() const { return outer_size_; }
  inline size_t AxisSize() const { return axis_size_; }
  inline size_t InnerSize() const { return inner_size_; }

 private:
  size_t outer_size_{0};
  size_t axis_size_{0};
  size_t inner_size_{0};
  size_t axis_offset_{0};
};

template <size_t Ndim>
class NdTensorIterator {
 public:
  template <typename... Indexes>
  NdTensorIterator(int64_t first_dim, Indexes... rest_dims)
      : dims_{{first_dim, rest_dims...}}, size_{(first_dim * ... * rest_dims)} {
    static_assert(sizeof...(rest_dims) + 1 == Ndim, "Input dimensions should match Ndim");
  }

  template <typename... Indexes>
  int64_t operator()(const Indexes... dims) const {
    static_assert(sizeof...(dims) == Ndim, "Input dimensions should match Ndim");
    return CalIndex(0, dims...);
  }

  template <typename... Indexes>
  int64_t at(const Indexes... dims) const {
    static_assert(sizeof...(dims) == Ndim, "Input dimensions should match Ndim");
    const int64_t index = CalIndex<true>(0, dims...);
    if (index > size_) {
      MS_LOG(ERROR) << "Pos " << index << " is larger than array size " << size_;
    }
    return index;
  }

 private:
  template <bool CheckParam = false, typename... Indexes>
  int64_t CalIndex(const int64_t sum, const int64_t first_dim, const Indexes... rest_dims) const {
    constexpr auto n = Ndim - sizeof...(rest_dims);
    if constexpr (CheckParam) {
      if (first_dim >= std::get<n - 1>(dims_)) {
        MS_LOG(ERROR) << "Error on index " << (n - 1) << ", " << first_dim << " should be lower than "
                      << std::get<n - 1>(dims_);
      }
    }
    return CalIndex<CheckParam>((sum + first_dim) * std::get<n>(dims_), rest_dims...);
  }

  template <bool CheckParam = false>
  int64_t CalIndex(const int64_t sum, const int64_t first_dim) const {
    if constexpr (CheckParam) {
      if (first_dim >= std::get<Ndim - 1>(dims_)) {
        MS_LOG(ERROR) << "Error on index " << (Ndim - 1) << ", " << first_dim << " should be lower than "
                      << std::get<Ndim - 1>(dims_);
      }
    }
    return sum + first_dim;
  }

  const std::array<int64_t, Ndim> dims_;
  const int64_t size_;
};
int Sign(float x);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CPU_KERNEL_H_
