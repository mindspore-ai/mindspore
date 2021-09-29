/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/kernel.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "ir/anf.h"
#include "runtime/framework/graph_scheduler.h"
#include "actor/actormgr.h"
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_IX86) || defined(_M_X64)
#define PLATFORM_86
#endif

using mindspore::kernel::Address;
using mindspore::kernel::AddressPtr;
using CTask = std::function<void(size_t, size_t)>;
namespace mindspore {
namespace kernel {
constexpr char KERNEL_SIZE[] = "kernel_size";
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

struct ParallelSearchInfo {
  double min_cost_time{DBL_MAX};
  double tmp_sum_cost_time{0};
  float best_block_size;
  size_t best_pow{0};
  size_t search_count{0};
};

class CPUKernel : public kernel::KernelMod {
 public:
  CPUKernel() = default;
  ~CPUKernel() override = default;
  virtual void Init(const CNodePtr &kernel_node);
  virtual void InitKernel(const CNodePtr &kernel_node) = 0;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void * /*stream_ptr*/) override {
    return Launch(inputs, workspace, outputs);
  };
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs) = 0;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

 protected:
  virtual void InitInputOutputSize(const CNodePtr &kernel_node);
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  ParallelSearchInfo parallel_search_info_;

  template <typename T>
  inline T *GetDeviceAddress(const std::vector<AddressPtr> &addr_list, size_t index) {
    if (index >= addr_list.size()) {
      MS_LOG(EXCEPTION) << "Address index(" << index << ") out of range(" << addr_list.size() << ")";
    }

    if ((addr_list[index] == nullptr) || (addr_list[index]->addr == nullptr) || (addr_list[index]->size == 0)) {
      MS_LOG(EXCEPTION) << "The device address is empty, address index: " << index;
    }

    return reinterpret_cast<T *>(addr_list[index]->addr);
  }
};

class CPUKernelUtils {
 public:
  static void ExpandDimsTo4(std::vector<size_t> *shape);
  static size_t CalcOffset(const std::vector<size_t> &shape, size_t dim0, size_t dim1, size_t dim2, size_t dim3);
  static size_t GetElementNumOnAxis(const std::vector<size_t> &shape, int axis);
  static void GetElementNumEveryDim(const std::vector<size_t> &shape, std::vector<size_t> *element_num);
  static void ParallelFor(const CTask &task, size_t count, float block_size = 128.0);
  static std::vector<size_t> FlatShapeByAxis(const std::vector<size_t> &shape, int axis);
  static std::vector<size_t> GetBroadcastShape(const std::vector<size_t> &x, const std::vector<size_t> &y);
  static void ParallelForAutoSearch(const CTask &task, size_t count, ParallelSearchInfo *parallel_search_info);
};

class BroadcastIterator {
 public:
  BroadcastIterator(std::vector<size_t> input_shape_a, std::vector<size_t> input_shape_b,
                    std::vector<size_t> output_shape);
  virtual ~BroadcastIterator() = default;
  inline size_t GetInputPosA() const { return input_pos_[0]; }
  inline size_t GetInputPosB() const { return input_pos_[1]; }
  void SetPos(size_t pos);
  void GenNextPos();

 private:
  void BroadcastShape();
  void InitStrides();

  std::vector<size_t> coordinates_;
  std::vector<size_t> input_shape_a_;
  std::vector<size_t> input_shape_b_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> input_strides_a_;
  std::vector<size_t> input_strides_b_;
  std::vector<size_t> input_back_strides_a_;
  std::vector<size_t> input_back_strides_b_;
  std::array<size_t, 2> input_pos_{0};
  int output_dimension_{0};
};

class TransposeIterator {
 public:
  TransposeIterator(std::vector<size_t> output_shape, std::vector<size_t> axes, const std::vector<size_t> &input_shape);
  virtual ~TransposeIterator() = default;
  inline size_t GetPos() const { return pos_; }
  void SetPos(size_t pos);
  void GenNextPos();

 private:
  int dimension_{0};
  std::vector<size_t> coordinates_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  std::vector<size_t> back_strides_;
  std::vector<size_t> axes_;
  size_t pos_{0};
};

ActorThreadPool *GetActorMgrInnerThreadPool();
void ParallelLaunch(const CTask &task, size_t count, float block_size = 128.0, Content content = nullptr);
void ParallelLaunchAutoSearch(const CTask &task, size_t count, Content content,
                              ParallelSearchInfo *parallel_search_info);

class AxisIterator {
 public:
  AxisIterator() = default;
  virtual ~AxisIterator() = default;
  void Init(const std::vector<size_t> &input_shape, size_t axis);

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
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CPU_KERNEL_H_
