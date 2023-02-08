/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_EINSUM_GRAD_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_EINSUM_GRAD_KERNEL_H_

#include <ctype.h>
#include <vector>
#include <string>
#include <set>
#include <unordered_map>
#include <functional>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/math/einsum_helper.h"
namespace mindspore {
namespace kernel {
constexpr int INPUT_NUM_MIN = 2;
constexpr int IDX_NAME = 0;
constexpr int IDX_INP_SHAPE = 1;
constexpr int IDX_PARAM = 2;
constexpr int IDX_OUT_SHAPE = 3;
template <typename T>
class EinsumGradGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  EinsumGradGpuKernelMod() { ResetResource(); }
  ~EinsumGradGpuKernelMod() = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  // inputs: 0, data0; 1, data1;...k,datak; k+1, dout
  // workspace: 0, mid_res0; 1,mid_res1; ... k-1,mid_resk-1; k,work0; k + 1,work1; k+2, shape0;k+3; shape1,k+4; shape2,
  // outputs: 0 dinput0; 1dinput1; ... k, dinputk
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    std::vector<void *> workspace_ptr_list(workspace.size());
    for (size_t idx = 0; idx < workspace.size(); ++idx) {
      workspace_ptr_list[idx] = GetDeviceAddress<void>(workspace, idx);
    }
    func_helper_.Init(workspace_ptr_list, type_id_);
    if (inputs.size() > INPUT_NUM_MIN) {
      LaunchForward(inputs, workspace, outputs, stream_ptr);
    } else {
      size_t size = sizeof(T);
      for (auto &dim : out_shape_) {
        size *= dim;
      }
      T *src_ptr = GetDeviceAddress<T>(inputs, 1);
      T *dst_ptr = GetDeviceAddress<T>(outputs, 0);
      CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
        cudaMemcpyAsync(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
        "For " + node_name_ + ", cudaMemcpyAsync failed.");
    }
    LaunchBackward(inputs, workspace, outputs, stream_ptr);
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto node_name = common::AnfAlgo::GetCNodeName(kernel_node);
    node_name_ = node_name;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num < INPUT_NUM_MIN) {
      MS_LOG(ERROR) << "For " << node_name_ << ", input number should be no less than 2, but got " << input_num;
      return false;
    }
    type_id_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
    for (size_t idx = 0; idx < input_num - 1; ++idx) {
      TypeId cur_type_id = AnfAlgo::GetInputDeviceDataType(kernel_node, idx);
      if (cur_type_id != type_id_) {
        MS_LOG(ERROR) << "For " << node_name_ << ", input types should be the same, but it does not.";
        return false;
      }
      auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, idx);
      input_shapes_.push_back(in_shape);
    }
    std::string equation = GetAttr<std::string>(kernel_node, "equation");
    single_op_ = std::vector<std::vector<OpStruct>>(input_shapes_.size());
    bool flag = func_helper_.Preprocess(equation, node_name, input_shapes_, &out_shape_, &single_op_, &res_op_);
    if (!flag) {
      return false;
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    input_shapes_.clear();
    out_shape_.clear();
    single_op_.clear();
    res_op_.clear();
    reduce_sum_wrok_size_ = 0;
    work_size_ = 0;
    shape_size_ = 0;
  }
  // inputs: (0, data0; 1, data1;...k,datak); k+1, dout
  // workspace: 0, mid_res0; 1,mid_res1; ... k-1,mid_resk-1; k,work0; k + 1,work1; k+2, shape0;k+3; shape1,k+4; shape2,
  // outputs: 0 dinput0; 1dinput1; ... k, dinputk
 protected:
  void InitSizeLists() override {
    InitInpOutSizeLists();
    InitWorkSizeLists();
  }
  void InitInpOutSizeLists() {
    size_t size = 0;
    size_t mul_val = (type_id_ == kNumberTypeFloat16) ? HALF_TYPE_WORK_SIZE_MUL : 1;
    for (auto &op_vec : single_op_) {
      auto &inp_shape = std::get<IDX_INP_SHAPE>(op_vec[0]);
      size = func_helper_.GetShapeSize(inp_shape);
      work_size_ = work_size_ > size ? work_size_ : size;
      shape_size_ = inp_shape.size() > shape_size_ ? inp_shape.size() : shape_size_;
      input_size_list_.emplace_back(size);
      output_size_list_.emplace_back(size);

      for (auto &cur_op : op_vec) {
        auto name = std::get<IDX_NAME>(cur_op);
        auto shape = std::get<IDX_INP_SHAPE>(cur_op);
        size = func_helper_.GetShapeSize(shape);
        if (name == "ReduceSum") {
          reduce_sum_wrok_size_ = reduce_sum_wrok_size_ > size * mul_val ? reduce_sum_wrok_size_ : size * mul_val;
        }
      }
    }
    size = func_helper_.GetShapeSize(out_shape_);
    input_size_list_.emplace_back(size);
  }
  void InitWorkSizeLists() {
    size_t size = 0;
    size_t mul_val = (type_id_ == kNumberTypeFloat16) ? HALF_TYPE_WORK_SIZE_MUL : 1;
    size_t mul_wrok_size = 0;
    for (auto &op : res_op_) {
      auto name = std::get<IDX_NAME>(op);
      auto out_shape = std::get<IDX_OUT_SHAPE>(op);
      size = func_helper_.GetShapeSize(out_shape);
      if (name == "Mul") {
        mul_wrok_size = mul_wrok_size > size ? mul_wrok_size : size;
        reduce_sum_wrok_size_ = reduce_sum_wrok_size_ > size * mul_val ? reduce_sum_wrok_size_ : size * mul_val;
      } else if (name == "ReduceSum") {
        auto inp_shape = std::get<IDX_INP_SHAPE>(op);
        size = func_helper_.GetShapeSize(inp_shape);
        reduce_sum_wrok_size_ = reduce_sum_wrok_size_ > size * mul_val ? reduce_sum_wrok_size_ : size * mul_val;
      }
      if (single_op_func_.count(name) == 0) {
        shape_size_ = out_shape.size() > shape_size_ ? out_shape.size() : shape_size_;
        work_size_ = work_size_ > size ? work_size_ : size;
        workspace_size_list_.emplace_back(size);
      }
    }
    workspace_size_list_.emplace_back(work_size_);
    workspace_size_list_.emplace_back(work_size_);
    if (mul_wrok_size > 0) {
      workspace_size_list_.emplace_back(mul_wrok_size);
    }
    if (reduce_sum_wrok_size_ > 0) {
      workspace_size_list_.emplace_back(reduce_sum_wrok_size_);
    }
    workspace_size_list_.emplace_back(shape_size_ * sizeof(size_t));
    workspace_size_list_.emplace_back(shape_size_ * sizeof(size_t));
    workspace_size_list_.emplace_back(shape_size_ * sizeof(size_t));
  }
  void LaunchForward(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    SingleOpForward(inputs, workspace, outputs, stream_ptr);
    ResOpForward(inputs, workspace, outputs, stream_ptr);
  }
  // inputs: 0, data0; 1, data1;...k,datak; k+1, dout
  // workspace: 0, mid_res0; 1,mid_res1; ... k-1,mid_resk-1; k,work0; k + 1,work1; k+2, shape0;k+3; shape1,k+4; shape2,
  // outputs: 0 dinput0; 1dinput1; ... k, dinputk
  void LaunchBackward(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    ResOpBackward(inputs, workspace, outputs, stream_ptr);
    SingleOpBackward(inputs, workspace, outputs, stream_ptr);
  }
  void ResOpBackward(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    size_t max_x_idx = inputs.size() - DIM_TWO;
    T *dout = GetDeviceAddress<T>(inputs, max_x_idx + 1);
    int idx_op = res_op_.size() - 1;
    T *src_ptr = dout;
    T *work0 = GetDeviceAddress<T>(workspace, max_x_idx);
    T *work1 = GetDeviceAddress<T>(workspace, max_x_idx + 1);
    T *dst_ptr = work0;
    int two_op_cnt = max_x_idx - 1;
    while (idx_op >= 0) {
      auto name = std::get<IDX_NAME>(res_op_[idx_op]);
      auto shape_a = Convert2SizeTClipNeg(std::get<IDX_INP_SHAPE>(res_op_[idx_op]));
      auto shape_b = std::get<IDX_PARAM>(res_op_[idx_op]);
      auto shape_c = Convert2SizeTClipNeg(std::get<IDX_OUT_SHAPE>(res_op_[idx_op]));
      if (single_op_func_.count(name) != 0) {
        func_helper_.SingleElementProcessGrad(name, src_ptr, dst_ptr, shape_a, shape_b, stream_ptr);
        if (dst_ptr == work0) {
          src_ptr = dst_ptr;
          dst_ptr = work1;
        } else {
          dst_ptr = src_ptr;
          src_ptr = work1;
        }
        --idx_op;
        if (idx_op >= 0) {
          name = std::get<IDX_NAME>(res_op_[idx_op]);
          while (idx_op >= 0 && single_op_func_.count(name) != 0) {
            shape_a = Convert2SizeTClipNeg(std::get<IDX_INP_SHAPE>(res_op_[idx_op]));
            shape_b = std::get<IDX_PARAM>(res_op_[idx_op]);
            shape_c = Convert2SizeTClipNeg(std::get<IDX_OUT_SHAPE>(res_op_[idx_op]));
            func_helper_.SingleElementProcessGrad(name, src_ptr, dst_ptr, shape_a, shape_b, stream_ptr);
            --idx_op;
            T *temp_ptr = dst_ptr;
            dst_ptr = src_ptr;
            src_ptr = temp_ptr;
            if (idx_op >= 0) {
              name = std::get<IDX_NAME>(res_op_[idx_op]);
            }
          }
        }
      }
      name = std::get<IDX_NAME>(res_op_[idx_op]);
      shape_a = Convert2SizeTClipNeg(std::get<IDX_INP_SHAPE>(res_op_[idx_op]));
      shape_b = std::get<IDX_PARAM>(res_op_[idx_op]);
      shape_c = Convert2SizeTClipNeg(std::get<IDX_OUT_SHAPE>(res_op_[idx_op]));
      T *mid_res;
      if (two_op_cnt == 0) {
        mid_res = GetDeviceAddress<T>(outputs, 0);
      } else {
        mid_res = GetDeviceAddress<T>(workspace, two_op_cnt - 1);
      }
      T *dst_ptr_2 = GetDeviceAddress<T>(outputs, two_op_cnt + 1);

      func_helper_.TwoElementProcessGrad(name, src_ptr, mid_res, dst_ptr_2, dst_ptr, shape_a, shape_b, shape_c,
                                         stream_ptr);
      if (two_op_cnt == 0) {
        auto shape = input_shapes_[0];
        size_t size = func_helper_.GetShapeSize(shape);
        CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
          cudaMemcpyAsync(mid_res, dst_ptr, size, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
          "For " + node_name_ + ", cudaMemcpyAsync failed.");
      }
      if (src_ptr == dout && dst_ptr == work0) {
        src_ptr = work0;
        dst_ptr = work1;
      } else {
        T *temp_ptr = dst_ptr;
        dst_ptr = src_ptr;
        src_ptr = temp_ptr;
      }
      --idx_op;
      --two_op_cnt;
    }
  }
  void SingleOpBackward(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                        const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    size_t max_x_idx = inputs.size() - DIM_TWO;
    T *work0 = GetDeviceAddress<T>(workspace, max_x_idx);
    for (int idx_op = single_op_.size() - 1; idx_op >= 0; --idx_op) {
      T *out_ptr = GetDeviceAddress<T>(outputs, idx_op);
      T *src_ptr = out_ptr;
      T *dst_ptr = work0;
      for (int idx = single_op_[idx_op].size() - 1; idx >= 0; --idx) {
        auto name = std::get<IDX_NAME>(single_op_[idx_op][idx]);
        auto inp_shape = Convert2SizeTClipNeg(std::get<IDX_INP_SHAPE>(single_op_[idx_op][idx]));
        auto op_param = std::get<IDX_PARAM>(single_op_[idx_op][idx]);
        auto dout_shape = Convert2SizeTClipNeg(std::get<IDX_OUT_SHAPE>(single_op_[idx_op][idx]));
        func_helper_.SingleElementProcessGrad(name, src_ptr, dst_ptr, inp_shape, op_param, stream_ptr);
        T *temp = src_ptr;
        src_ptr = dst_ptr;
        dst_ptr = temp;
      }
      if (src_ptr != out_ptr) {
        auto back_shape = std::get<IDX_INP_SHAPE>(single_op_[idx_op][0]);
        size_t size = func_helper_.GetShapeSize(back_shape);
        CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
          cudaMemcpyAsync(out_ptr, src_ptr, size, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
          "For " + node_name_ + ", cudaMemcpyAsync failed.");
      }
    }
  }
  void SingleOpForward(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                       const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    // 正向
    size_t work_idx = inputs.size() - DIM_TWO;
    T *work0 = GetDeviceAddress<T>(workspace, work_idx);
    T *src_ptr = work0;
    for (size_t idx_op = 0; idx_op < single_op_.size(); ++idx_op) {
      T *inp_ptr = GetDeviceAddress<T>(inputs, idx_op);
      T *out_ptr = GetDeviceAddress<T>(outputs, idx_op);
      T *dst_ptr = out_ptr;
      auto name = std::get<IDX_NAME>(single_op_[idx_op][0]);
      auto inp_shape = Convert2SizeTClipNeg(std::get<IDX_INP_SHAPE>(single_op_[idx_op][0]));
      auto op_param = std::get<IDX_PARAM>(single_op_[idx_op][0]);
      func_helper_.SingleElementProcess(name, inp_ptr, dst_ptr, inp_shape, op_param, stream_ptr);
      src_ptr = dst_ptr;
      dst_ptr = work0;
      for (size_t idx = 1; idx < single_op_[idx_op].size(); ++idx) {
        name = std::get<IDX_NAME>(single_op_[idx_op][idx]);
        inp_shape = Convert2SizeTClipNeg(std::get<IDX_INP_SHAPE>(single_op_[idx_op][idx]));
        op_param = std::get<IDX_PARAM>(single_op_[idx_op][idx]);
        func_helper_.SingleElementProcess(name, src_ptr, dst_ptr, inp_shape, op_param, stream_ptr);
        T *temp = src_ptr;
        src_ptr = dst_ptr;
        dst_ptr = temp;
      }
      if (src_ptr != out_ptr) {
        auto back_shape = std::get<IDX_OUT_SHAPE>(single_op_[idx_op].back());
        size_t size = func_helper_.GetShapeSize(back_shape);
        CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
          cudaMemcpyAsync(out_ptr, src_ptr, size, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
          "For " + node_name_ + ", cudaMemcpyAsync failed.");
      }
    }
  }
  void ResOpForward(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    T *single_fir_ptr = GetDeviceAddress<T>(outputs, 0);
    T *middle_res_ptr = GetDeviceAddress<T>(workspace, 0);
    T *src_ptr = single_fir_ptr;
    T *dst_ptr = middle_res_ptr;
    size_t idx_op = 0;
    size_t two_op_cnt = 0;
    size_t work_idx = inputs.size() - DIM_TWO;
    T *work0 = GetDeviceAddress<T>(workspace, work_idx);
    while (idx_op < res_op_.size()) {
      auto name = std::get<IDX_NAME>(res_op_[idx_op]);
      auto shape_a = Convert2SizeTClipNeg(std::get<IDX_INP_SHAPE>(res_op_[idx_op]));
      auto shape_b = std::get<IDX_PARAM>(res_op_[idx_op]);
      auto shape_c = Convert2SizeTClipNeg(std::get<IDX_OUT_SHAPE>(res_op_[idx_op]));
      T *src_ptr_2 = GetDeviceAddress<T>(outputs, two_op_cnt + 1);
      func_helper_.TwoElementProcess(name, src_ptr, src_ptr_2, dst_ptr, shape_a, shape_b, shape_c, stream_ptr);
      ++idx_op;
      ++two_op_cnt;
      src_ptr = dst_ptr;
      dst_ptr = work0;
      if (idx_op < res_op_.size()) {
        name = std::get<IDX_NAME>(res_op_[idx_op]);
        while (idx_op < res_op_.size() && single_op_func_.count(name) != 0) {
          auto shape_a = Convert2SizeTClipNeg(std::get<IDX_INP_SHAPE>(res_op_[idx_op]));
          auto shape_b = std::get<IDX_PARAM>(res_op_[idx_op]);
          func_helper_.SingleElementProcess(name, src_ptr, dst_ptr, shape_a, shape_b, stream_ptr);
          T *temp = src_ptr;
          src_ptr = dst_ptr;
          dst_ptr = temp;
          ++idx_op;
          if (idx_op < res_op_.size()) {
            name = std::get<IDX_NAME>(res_op_[idx_op]);
          }
        }
      }
      if (src_ptr != middle_res_ptr) {
        auto shape_c = std::get<IDX_OUT_SHAPE>(res_op_[idx_op - 1]);
        size_t size = func_helper_.GetShapeSize(shape_c);
        CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(middle_res_ptr, src_ptr, size, cudaMemcpyDeviceToDevice,
                                                          reinterpret_cast<cudaStream_t>(stream_ptr)),
                                          "For " + node_name_ + ", cudaMemcpyAsync failed.");
      }
      middle_res_ptr = GetDeviceAddress<T>(workspace, two_op_cnt);
      dst_ptr = middle_res_ptr;
    }
  }

 private:
  EinsumHelper<T> func_helper_;
  std::string node_name_;
  TypeId type_id_;
  size_t work_size_;
  size_t shape_size_;
  size_t reduce_sum_wrok_size_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<int64_t> out_shape_;
  std::vector<std::vector<OpStruct>> single_op_;
  std::vector<OpStruct> res_op_;
  std::set<std::string> single_op_func_ = {"ReduceSum", "Diagonal", "Transpose"};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_EINSUM_GRAD_KERNEL_H_
