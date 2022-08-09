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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BESSEL_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BESSEL_HELPER_H_

#include <map>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bessel_impl.cuh"

namespace mindspore {
namespace cukernel {
enum BesselOptype {
  BESSEL_OP_K0 = 0,
  BESSEL_OP_K0e = 1,
  BESSEL_OP_K1 = 2,
  BESSEL_OP_K1e = 3,
  BESSEL_OP_J0 = 4,
  BESSEL_OP_J1 = 5,
  BESSEL_OP_I0 = 6,
  BESSEL_OP_I0e = 7,
  BESSEL_OP_I1 = 8,
  BESSEL_OP_I1e = 9,
  BESSEL_OP_Y0 = 10,
  BESSEL_OP_Y1 = 11,
  BESSEL_OP_INVALID_TYPE = 12
};

static const std::map<std::string, BesselOptype> kBesselOpTypeMap = {
  {"BesselJ0", BESSEL_OP_J0}, {"BesselJ1", BESSEL_OP_J1},   {"BesselK0", BESSEL_OP_K0}, {"BesselK0e", BESSEL_OP_K0e},
  {"BesselK1", BESSEL_OP_K1}, {"BesselK1e", BESSEL_OP_K1e}, {"BesselI0", BESSEL_OP_I0}, {"BesselI0e", BESSEL_OP_I0e},
  {"BesselI1", BESSEL_OP_I1}, {"BesselI1e", BESSEL_OP_I1e}, {"BesselY0", BESSEL_OP_Y0}, {"BesselY1", BESSEL_OP_Y1}};

template <typename T>
class BesselHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit BesselHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
    bessel_op_type_ = BESSEL_OP_INVALID_TYPE;
  }

  virtual ~BesselHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    auto iter = kBesselOpTypeMap.find(kernel_name_);
    if (iter == kBesselOpTypeMap.end()) {
      MS_LOG(ERROR) << "For 'BesselOp', only support these types: "
                    << kernel::Map2Str<std::map, BesselOptype>(kBesselOpTypeMap) << " currently, but got "
                    << kernel_name_;
      return -1;
    }
    bessel_op_type_ = iter->second;
    constexpr size_t INPUT_NUM = 1;
    int flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    output_size_list_ = input_size_list_;
    is_null_input_ = (flag == 1);
    if (flag != 0) {
      return flag;
    }
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    static std::map<BesselOptype, std::function<void(const size_t, const T *, T *, const uint32_t, cudaStream_t)>>
      func_map = {{BESSEL_OP_J0, CalBesselJ0<T>},   {BESSEL_OP_J1, CalBesselJ1<T>},   {BESSEL_OP_K0, CalBesselK0<T>},
                  {BESSEL_OP_K0e, CalBesselK0e<T>}, {BESSEL_OP_K1, CalBesselK1<T>},   {BESSEL_OP_K1e, CalBesselK1e<T>},
                  {BESSEL_OP_I0, CalBesselI0<T>},   {BESSEL_OP_I0e, CalBesselI0e<T>}, {BESSEL_OP_I1, CalBesselI1<T>},
                  {BESSEL_OP_I1e, CalBesselI1e<T>}, {BESSEL_OP_Y0, CalBesselY0<T>},   {BESSEL_OP_Y1, CalBesselY1<T>}};
    auto iter = func_map.find(bessel_op_type_);
    if (iter != func_map.end()) {
      T *input_addr;
      T *output_addr;
      int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_addr);
      if (flag != 0) {
        return flag;
      }
      flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_addr);
      if (flag != 0) {
        return flag;
      }
      iter->second(input_size_list_[0] / sizeof(T), input_addr, output_addr, device_id_,
                   reinterpret_cast<cudaStream_t>(cuda_stream));
    } else {
      MS_LOG(ERROR) << "For 'BesselOp', only support these types: "
                    << kernel::Map2Str<std::map, BesselOptype>(kBesselOpTypeMap) << " currently, but got "
                    << kernel_name_;
      return -1;
    }
    return 0;
  }

 private:
  BesselOptype bessel_op_type_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BESSEL_HELPER_H_
