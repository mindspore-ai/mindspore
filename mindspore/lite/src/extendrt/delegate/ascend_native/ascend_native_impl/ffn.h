/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_FFN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_FFN_H_
#include <vector>
#include "acl/acl.h"
// #define ACL_MOEFFN
#ifdef ACL_MOEFFN
#include "aclnnop/aclnn_moe_ffn.h"
#endif
#include "utils/log_adapter.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/gemm.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/unary_op.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/param.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/hccl_adapter.h"

namespace mindspore::ascend_native {
template <typename T>
class Ffn {
  typedef struct {
    uint64_t workspace_size_ = 0;
    aclTensor *in_x_ = nullptr;
    aclTensor *in_y_ = nullptr;
    aclIntArray *expert_to_token_ = nullptr;
    aclOpExecutor *executor_ = nullptr;
  } FfnAclExecT;

 public:
  Ffn() = default;
  ~Ffn() {
    for (auto &item : vcollect_) {
      aclDestroyTensor(item);
    }
    vcollect_.clear();
    if (expert_to_token_arr_) free(expert_to_token_arr_);
    expert_to_token_arr_ = nullptr;
    CleanupExecuter();
  }

  void Prepare(void *map_weights, void *map_bias, void *proj_weights, void *proj_bias, EncoderParams *p, int vcores,
               int ccores) {
    vcores_ = vcores;
    int expert_number = p->expert_num_;
    std::vector<int64_t> shape_w1;
    std::vector<int64_t> stride_w1;
    std::vector<int64_t> shape_b1;
    std::vector<int64_t> stride_b1;
    std::vector<int64_t> shape_w2;
    std::vector<int64_t> stride_w2;
    std::vector<int64_t> shape_b2;
    std::vector<int64_t> stride_b2;

    shape_w1 = {expert_number, p->hid_dim_, p->ffn_hid_dim_};
    shape_b1 = {expert_number, p->ffn_hid_dim_};
    shape_w2 = {expert_number, p->ffn_hid_dim_, p->hid_dim_};
    shape_b2 = {expert_number, p->hid_dim_};
    stride_w1 = calcStride(shape_w1);
    stride_b1 = calcStride(shape_b1);
    stride_w2 = calcStride(shape_w2);
    stride_b2 = calcStride(shape_b2);
    map_weights_ = map_weights;
    map_bias_ = map_bias;
    proj_weights_ = proj_weights;
    proj_bias_ = proj_bias;

    w1_tensor_ = aclCreateTensor(shape_w1.data(), shape_w1.size(), aclDataType::ACL_FLOAT16, stride_w1.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, shape_w1.data(), shape_w1.size(), map_weights);
    b1_tensor_ = aclCreateTensor(shape_b1.data(), shape_b1.size(), aclDataType::ACL_FLOAT16, stride_b1.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, shape_b1.data(), shape_b1.size(), map_bias);
    w2_tensor_ = aclCreateTensor(shape_w2.data(), shape_w2.size(), aclDataType::ACL_FLOAT16, stride_w2.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, shape_w2.data(), shape_w2.size(), proj_weights);
    b2_tensor_ = aclCreateTensor(shape_b2.data(), shape_b2.size(), aclDataType::ACL_FLOAT16, stride_b2.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, shape_b2.data(), shape_b2.size(), proj_bias);
    vcollect_ = {w1_tensor_, b1_tensor_, w2_tensor_, b2_tensor_};
    expert_to_token_arr_ = reinterpret_cast<int64_t *>(malloc(sizeof(int64_t) * (expert_number)));
    if (expert_to_token_arr_ == nullptr) {
      MS_LOG(ERROR) << "aclnnMoeFFN failed in prepare to malloc: " << sizeof(int64_t) * (expert_number);
    }
  }

  size_t GetWsSize(EncoderParams *p) { return p->batch_size_ * p->seq_ * p->ffn_hid_dim_ * sizeof(T); }

#ifdef ACL_MOEFFN
  int ComputeAclFuse(void *input, void *output, EncoderParams *p, void *ws, void *sys_ws, void *stream) {
    PrepareExecuter(p, input, output);
    auto ret = aclnnMoeFFN(sys_ws, ffn_exec_.workspace_size_, ffn_exec_.executor_, stream);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "aclnnMoeFFN failure: " << aclGetRecentErrMsg();
      return ret;
    }
    return ACL_SUCCESS;
  }
#endif
  int Compute(void *input_base, void *output_base, EncoderParams *p, void *ws, void *sys_ws, void *stream,
              void *alt_stream) {
    int offset_w = p->ffn_hid_dim_ * p->hid_dim_;
    int offset_b1 = p->ffn_hid_dim_;
    int offset_b2 = p->hid_dim_;
    int in_out_offset = 0;
    for (int e = 0; e < p->expert_num_; e++) {
      int token_num = p->expert_to_tokens_[e];
      if (token_num == 0) continue;
      // Setup expert
      T *map_weights = reinterpret_cast<T *>(map_weights_) + e * offset_w;
      T *map_bias = reinterpret_cast<T *>(map_bias_) + e * offset_b1;
      T *proj_weights = reinterpret_cast<T *>(proj_weights_) + e * offset_w;
      T *proj_bias = reinterpret_cast<T *>(proj_bias_) + e * offset_b2;
      T *input = reinterpret_cast<T *>(input_base) + in_out_offset * p->hid_dim_;
      T *output = reinterpret_cast<T *>(output_base) + in_out_offset * p->hid_dim_;
      // Do FFN
      gemm0_.execute(1, token_num, p->ffn_hid_dim_, p->hid_dim_, input, map_weights, ws, sys_ws, SYS_WS_RESERVED,
                     stream, map_bias);
      kernelFastGelu(ws, ws, token_num * p->ffn_hid_dim_, vcores_, stream);
      gemm1_.execute(1, token_num, p->hid_dim_, p->ffn_hid_dim_, ws, proj_weights, output, sys_ws, SYS_WS_RESERVED,
                     stream, proj_bias);
      in_out_offset += token_num;
    }
    if ((p->rank_num_ > 1)) {
      auto &hccl = HcclAdapter::GetInstance();
      uint64_t count = p->token_num_to_expert_ * p->hid_dim_;
      hccl.AllSumReduce(output_base, output_base, count, stream);
    }
    return ACL_SUCCESS;
  }

  int ComputeReduce(void *input_base, void *output_base, EncoderParams *p, void *ws, void *sys_ws, void *alt_stream,
                    void *stream) {
    int offset_w = p->ffn_hid_dim_ * p->hid_dim_;
    int offset_b1 = p->ffn_hid_dim_;
    int offset_b2 = p->hid_dim_;
    int in_out_offset = 0;
    for (int e = 0; e < p->expert_num_; e++) {
      int token_num = p->expert_to_tokens_[e];
      if (token_num == 0) continue;
      // setup expert
      T *map_weights = reinterpret_cast<T *>(map_weights_) + e * offset_w;
      T *map_bias = reinterpret_cast<T *>(map_bias_) + e * offset_b1;
      T *proj_weights = reinterpret_cast<T *>(proj_weights_) + e * offset_w;
      T *proj_bias = reinterpret_cast<T *>(proj_bias_) + e * offset_b2;
      T *input = reinterpret_cast<T *>(input_base) + in_out_offset * p->hid_dim_;
      T *output = reinterpret_cast<T *>(output_base) + in_out_offset * p->hid_dim_;

      Gemm gemm;
      GemmDistrubute gemmd;
      gemm.execute(1, token_num, p->ffn_hid_dim_, p->hid_dim_, input, map_weights, ws, sys_ws, SYS_WS_RESERVED, stream,
                   map_bias);
      kernelFastGelu(ws, ws, token_num * p->ffn_hid_dim_, vcores_, stream);
      gemmd.execute(token_num, p->hid_dim_, p->ffn_hid_dim_, 2, ws, proj_weights, output, proj_bias, sys_ws,
                    SYS_WS_RESERVED, alt_stream, stream);
      if ((p->rank_num_ > 1) && (p->incremental_mode_ == false)) {
        auto &hccl = HcclAdapter::GetInstance();
        uint64_t count = token_num * p->hid_dim_;
        hccl.AllSumReduce(output, output, count, stream);
      }
      in_out_offset += token_num;
    }
    if ((p->rank_num_ > 1) && (p->incremental_mode_ == true)) {
      auto &hccl = HcclAdapter::GetInstance();
      uint64_t count = p->token_num_to_expert_ * p->hid_dim_;
      hccl.AllSumReduce(output_base, output_base, count, stream);
    }
    return ACL_SUCCESS;
  }

 private:
  void CleanupExecuter() {
    if (ffn_exec_.in_x_) {
      aclDestroyTensor(ffn_exec_.in_x_);
      ffn_exec_.in_x_ = nullptr;
    }
    if (ffn_exec_.in_y_) {
      aclDestroyTensor(ffn_exec_.in_y_);
      ffn_exec_.in_y_ = nullptr;
    }
    if (ffn_exec_.expert_to_token_) {
      aclDestroyIntArray(ffn_exec_.expert_to_token_);
      ffn_exec_.expert_to_token_ = nullptr;
    }
    ffn_exec_.executor_ = nullptr;
    return;
  }
#ifdef ACL_MOEFFN
  int PrepareExecuter(EncoderParams *p, void *input, void *output) {
    CleanupExecuter();
    int token_num = p->token_num_to_expert_;
    int expert_number = p->expert_num_;

    std::vector<int64_t> shape_xy = {token_num, p->hid_dim_};
    auto stride_xy = calcStride(shape_xy);

    ffn_exec_.in_x_ = aclCreateTensor(shape_xy.data(), shape_xy.size(), aclDataType::ACL_FLOAT16, stride_xy.data(), 0,
                                      aclFormat::ACL_FORMAT_ND, shape_xy.data(), shape_xy.size(), input);
    ffn_exec_.in_y_ = aclCreateTensor(shape_xy.data(), shape_xy.size(), aclDataType::ACL_FLOAT16, stride_xy.data(), 0,
                                      aclFormat::ACL_FORMAT_ND, shape_xy.data(), shape_xy.size(), output);
    for (int i = 0; i < expert_number; i++) {
      expert_to_token_arr_[i] = (int64_t)p->expert_to_tokens_[i];
    }

    ffn_exec_.expert_to_token_ = aclCreateIntArray(expert_to_token_arr_, expert_number);
    char activation[] = "fastgelu";
    auto ret = aclnnMoeFFNGetWorkspaceSize(ffn_exec_.in_x_, ffn_exec_.expert_to_token_, w1_tensor_, b1_tensor_,
                                           w2_tensor_, b2_tensor_, nullptr, nullptr, nullptr, nullptr, activation,
                                           ffn_exec_.in_y_, &ffn_exec_.workspace_size_, &ffn_exec_.executor_);
    if (ret != OK) {
      MS_LOG(ERROR) << "aclnnMoeFFNGetWorkspaceSize failed:" << aclGetRecentErrMsg();
      return ret;
    }
    if (ffn_exec_.workspace_size_ > SYS_WS_RESERVED) {
      MS_LOG(ERROR) << "aclnnMoeFFNGetWorkspaceSize too big:" << ffn_exec_.workspace_size_;
      return -1;
    }
    return ACL_SUCCESS;
  }
#endif

  int vcores_;
  std::vector<aclTensor *> vcollect_;
  aclTensor *w1_tensor_ = nullptr;
  aclTensor *b1_tensor_ = nullptr;
  aclTensor *w2_tensor_ = nullptr;
  aclTensor *b2_tensor_ = nullptr;
  int64_t *expert_to_token_arr_ = nullptr;
  FfnAclExecT ffn_exec_;
  Gemm gemm0_;
  Gemm gemm1_;
  void *map_weights_ = nullptr;
  void *map_bias_ = nullptr;
  void *proj_weights_ = nullptr;
  void *proj_bias_ = nullptr;
};
}  // namespace mindspore::ascend_native
#endif
