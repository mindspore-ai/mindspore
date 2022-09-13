/* Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SEGMENT_OPS_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SEGMENT_OPS_HELPER_H_

#include <map>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/segment_impl.cuh"

namespace mindspore {
namespace cukernel {
enum SegmentOpsOptype {
  SEGMENT_OP_MAX = 0,
  SEGMENT_OP_MIN = 1,
  SEGMENT_OP_MEAN = 2,
  SEGMENT_OP_SUM = 3,
  SEGMENT_OP_PROD = 4,
  SEGMENT_OP_INVALID_TYPE = 5
};

static const std::map<std::string, SegmentOpsOptype> kSegmentOpsOpTypeMap = {{"SegmentMax", SEGMENT_OP_MAX},
                                                                             {"SegmentMin", SEGMENT_OP_MIN},
                                                                             {"SegmentMean", SEGMENT_OP_MEAN},
                                                                             {"SegmentSum", SEGMENT_OP_SUM},
                                                                             {"SegmentProd", SEGMENT_OP_PROD}};
template <typename T, typename S>
class SegmentOpsHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit SegmentOpsHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
    Segment_ops_op_type_ = SEGMENT_OP_INVALID_TYPE;
  }

  virtual ~SegmentOpsHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    auto iter = kSegmentOpsOpTypeMap.find(kernel_name_);
    if (iter == kSegmentOpsOpTypeMap.end()) {
      MS_LOG(ERROR) << "For 'SegmentOps', only support these types: "
                    << kernel::Map2Str<std::map, SegmentOpsOptype>(kSegmentOpsOpTypeMap) << " currently, but got "
                    << kernel_name_;
      return -1;
    }
    Segment_ops_op_type_ = iter->second;
    constexpr size_t INPUT_NUM = 2;
    constexpr size_t OUTPUT_NUM = 1;
    int inp_flag = CalShapesNum(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);

    if (inp_flag == 1 || out_flag == 1) {
      is_null_input_ = true;
      return 0;
    }
    if (input_shapes[0].size() < 1) {
      MS_LOG(ERROR) << "For 'SegmentOps', data shape must be more than 1D, but got " << input_shapes[0].size();
      return -1;
    }
    if (input_shapes[1].size() != 1) {
      MS_LOG(ERROR) << "For 'SegmentOps', segment_ids' shape only support 1D, but got " << input_shapes[1].size();
      return -1;
    }
    outer_class_ = output_shapes[0][0];
    outer_size_ = input_shapes[0][0];
    inner_size_ = input_size_list_[0] / outer_size_;
    size_t segment_id_num = static_cast<size_t>(input_shapes[1][0]);
    if (segment_id_num != outer_size_) {
      MS_LOG(ERROR) << "For 'SegmentOps', the length of segment_id must be equal to input_shape[0],"
                       " but got the length of segment_id : "
                    << segment_id_num << ", and input_shape[0] " << outer_size_;
      return -1;
    }
    input_size_list_[0] *= sizeof(T);
    input_size_list_[1] *= sizeof(S);
    work_size_list_.emplace_back((outer_size_ + 1) * sizeof(size_t));
    return 0;
  }

  static int CalShapesNum(const std::vector<std::vector<int64_t>> &shapes, const size_t shape_num,
                          const std::string kernel_name, const std::string param_name,
                          std::vector<size_t> *shapes_size) {
    if (shape_num != shapes.size()) {
      MS_LOG(ERROR) << "For '" << kernel_name << "', the number of " << param_name << "should be equal to " << shape_num
                    << ", but got " << shapes.size();
      return -1;
    }
    int return_flag = 0;
    for (size_t idx = 0; idx < shape_num; ++idx) {
      size_t cur_size = 1;
      if (shapes[idx].size() == 0) {
        MS_LOG(WARNING) << "For '" << kernel_name << "', the shapes[" << idx << "] is ( )";
        shapes_size->emplace_back(cur_size);
        continue;
      }
      for (const auto &val : shapes[idx]) {
        cur_size *= val;
      }
      if (cur_size == 0) {
        MS_LOG(WARNING) << "For '" << kernel_name << "', got shapes[" << idx << "] is "
                        << ConvertVectorToString(shapes[idx]);
        return_flag = 1;
      }
      shapes_size->emplace_back(cur_size);
    }
    return return_flag;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *input_addr;
    S *seg_id_addr;
    size_t *seg_pos_addr;
    T *output_addr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_addr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(input_ptrs, 1, kernel_name_, &seg_id_addr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<size_t>(work_ptrs, 0, kernel_name_, &seg_pos_addr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_addr);
    if (flag != 0) {
      return flag;
    }
    CalSegmentCombination(input_addr, output_addr, seg_id_addr, seg_pos_addr, Segment_ops_op_type_, inner_size_,
                          outer_size_, outer_class_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void ResetResource() noexcept override {
    inner_size_ = 1;
    outer_size_ = 1;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

 private:
  SegmentOpsOptype Segment_ops_op_type_;
  size_t inner_size_;
  size_t outer_size_;
  size_t outer_class_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SEGMENT_OPS_HELPER_H_
