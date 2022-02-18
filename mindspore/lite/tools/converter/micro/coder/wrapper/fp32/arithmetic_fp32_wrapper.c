/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "wrapper/fp32/arithmetic_fp32_wrapper.h"
void TileConstTensor(const float *in_data, float *out_data, size_t ndim, const int *in_shape, const int *in_strides,
                     const int *out_strides, const int *multiple) {
  TileOneDimensionFp32(in_data, out_data, 0, ndim, in_shape, in_strides, out_strides, multiple);
}

void ArithmeticExecute(const void *input0, const void *input1, void *output, int size, bool is_opt,
                       ArithmeticFuncType arithmetic_func_type, const void *arithmetic_func,
                       const ArithmeticParameter *param) {
  if (arithmetic_func_type == kArithmeticFuncFloat) {
    if (is_opt) {
      ArithmeticOptRun arithmetic_opt_run = (ArithmeticOptRun)(arithmetic_func);
      arithmetic_opt_run((const float *)(input0), (const float *)(input1), (float *)(output), size, param);
    } else {
      ArithmeticRun arithmetic_run = (ArithmeticRun)(arithmetic_func);
      arithmetic_run((const float *)(input0), (const float *)(input1), (float *)(output), size);
    }
  } else if (arithmetic_func_type == kArithmeticFuncBool) {
    ArithmeticBoolRun arithmetic_run_bool = (ArithmeticBoolRun)(arithmetic_func);
    arithmetic_run_bool((const bool *)(input0), (const bool *)(input1), (bool *)(output), size);
  } else if (arithmetic_func_type == kArithmeticFuncInt) {
    if (is_opt) {
      ArithmeticOptIntRun arithmetic_opt_run_int = (ArithmeticOptIntRun)(arithmetic_func);
      arithmetic_opt_run_int((const int *)(input0), (const int *)(input1), (int *)(output), size, param);
    } else {
      ArithmeticIntRun arithmetic_run_int = (ArithmeticIntRun)(arithmetic_func);
      arithmetic_run_int((const int *)(input0), (const int *)(input1), (int *)(output), size);
    }
  }
}

void BatchScalarCalc(const void *input0, const void *input1, void *output, int batch_size, int size, bool is_opt,
                     const void *arithmetic_func, const ArithmeticWrapperInfo *wrapper_info,
                     const ArithmeticParameter *param) {
  int offset0 = wrapper_info->offset0_;
  int offset1 = wrapper_info->offset1_;
  int out_offset = wrapper_info->out_offset_;
  int stride0 = wrapper_info->stride0_;
  int stride1 = wrapper_info->stride1_;
  int out_stride = wrapper_info->out_stride_;
  for (int i = 0; i < batch_size; i++) {
    ArithmeticExecute((const uint8_t *)(input0) + offset0, (const uint8_t *)(input1) + offset1,
                      (uint8_t *)(output) + out_offset, size, is_opt, wrapper_info->arithmetic_func_type_,
                      arithmetic_func, param);
    offset0 += stride0;
    offset1 += stride1;
    out_offset += out_stride;
  }
}

void BroadcastRun(const void *input0, const void *input1, void *output, int dim, int out_count, int out_thread_stride,
                  int break_pos, int data_type_len, ArithmeticFuncType arithmetic_func_type,
                  const void *arithmetic_func, const ArithmeticParameter *param) {
  if (dim > break_pos) {
    int offset = out_thread_stride * data_type_len;
    ArithmeticExecute((const uint8_t *)(input0) + offset, (const uint8_t *)(input1) + offset,
                      (uint8_t *)(output) + offset, out_count, false, arithmetic_func_type, arithmetic_func, param);
  }
  int offset_size[] = {param->in_strides0_[dim] * data_type_len, param->in_strides1_[dim] * data_type_len,
                       param->out_strides_[dim] * data_type_len};
  for (int i = 0; i < param->out_shape_[dim]; ++i) {
    int pos0_ = param->in_shape0_[dim] == 1 ? 0 : i;
    int pos1_ = param->in_shape1_[dim] == 1 ? 0 : i;
    BroadcastRun((const uint8_t *)(input0) + pos0_ * offset_size[0], (const uint8_t *)(input1) + pos1_ * offset_size[1],
                 (uint8_t *)(output) + i * offset_size[2], dim + 1, out_count, out_thread_stride, break_pos,
                 data_type_len, arithmetic_func_type, arithmetic_func, param);
  }
}
