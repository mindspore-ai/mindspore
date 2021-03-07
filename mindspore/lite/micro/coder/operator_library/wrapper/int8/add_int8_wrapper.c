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

#include "wrapper/int8/add_int8_wrapper.h"
#include "nnacl/errorcode.h"

int AddBroadcastInt8Run(void *cdata, int task_id) {
  AddInt8Args *args = (AddInt8Args *)(cdata);
  int stride = UP_DIV(args->out_size_, args->thread_count_);
  int real_out_count = MSMIN(stride, args->out_size_ - stride * task_id);
  if (real_out_count <= 0) {
    return NNACL_OK;
  }
  int8_t *cur_in0 = NULL;
  int8_t *cur_in1 = NULL;
  int8_t *cur_out = NULL;
  for (int i = 0; i < real_out_count; i++) {
    if (args->arith_para_->in_elements_num0_ == args->arith_para_->out_elements_num_) {
      cur_in0 = args->input0_data_ + task_id * stride * args->in_size_ + i * args->in_size_;
      cur_in1 = args->input1_data_;
      cur_out = args->output_data_ + task_id * stride * args->in_size_ + i * args->in_size_;
    } else {
      cur_in0 = args->input0_data_;
      cur_in1 = args->input1_data_ + task_id * stride * args->in_size_ + i * args->in_size_;
      cur_out = args->output_data_ + task_id * stride * args->in_size_ + i * args->in_size_;
    }
    AddInt8(cur_in0, cur_in1, cur_out, args->in_size_, args->para_);
  }
  return NNACL_OK;
}

int AddInt8Run(void *cdata, int task_id) {
  AddInt8Args *args = (AddInt8Args *)(cdata);
  /* no need broadcast */
  int stride = UP_DIV(args->elements_num_, args->thread_count_);
  int rest_count = args->elements_num_ - task_id * stride;
  int real_count = MSMIN(stride, rest_count);
  if (real_count <= 0) {
    return NNACL_OK;
  }
  int8_t *cur_in0 = args->input0_data_ + stride * task_id;
  int8_t *cur_in1 = args->input1_data_ + stride * task_id;
  int8_t *cur_out = args->output_data_ + stride * task_id;
  if (args->support_opt_add_) {
    int8_t *ptr_in = args->arith_para_->in_elements_num0_ == 1 ? cur_in1 : cur_in0;
    int8_t element_in = args->arith_para_->in_elements_num0_ == 1 ? args->input0_data_[0] : args->input1_data_[0];
    AddQuantQrgs *ptr_args =
      args->arith_para_->in_elements_num0_ == 1 ? &args->para_->in1_args_ : &args->para_->in0_args_;
    AddQuantQrgs *ele_args =
      args->arith_para_->in_elements_num0_ == 1 ? &args->para_->in0_args_ : &args->para_->in1_args_;
    AddOptInt8(ptr_in, element_in, cur_out, rest_count, args->para_, ptr_args, ele_args);
  } else {
    AddInt8(cur_in0, cur_in1, cur_out, rest_count, args->para_);
  }
  return NNACL_OK;
}
