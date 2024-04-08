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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_FLASH_ATTENTION_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_FLASH_ATTENTION_H_

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <limits>
#include "acl/acl.h"
#include "utils/log_adapter.h"
#include "aclnnop/aclnn_prompt_flash_attention.h"
#include "aclnnop/aclnn_incre_flash_attention.h"
#include "aclnnop/aclnn_permute.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/param.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/tiling_if.h"

#include "extendrt/delegate/ascend_native/ascend_native_impl/ai_core/matmul.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/gemm.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/encoder_vector_kernels.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/hccl_adapter.h"

namespace mindspore::ascend_native {

template <typename T>
class FlashAttn {
  typedef struct {
    uint64_t workspace_size_ = 0;
    aclTensor *q_ = nullptr;
    aclTensor *k_ = nullptr;
    aclTensor *v_ = nullptr;
    aclTensor *mask_ = nullptr;
    aclTensor *out_ = nullptr;
    aclTensorList *k_list_ = nullptr;
    aclTensorList *v_list_ = nullptr;
    aclIntArray *act_seq_ = nullptr;
    aclOpExecutor *executor_ = nullptr;
  } AttnAclExecT;

 public:
  FlashAttn() = default;

  void Prepare(void *q_w, void *kv_w, void *qkv_b, void *k_cache, void *v_cache, void *projection_w, void *projection_b,
               EncoderParams *p, int vcores, int ccores) {
    cube_cores_ = ccores;
    vec_cores_ = vcores;
    if (p->is_query_) {
      q_w_ = q_w;
      kv_w_ = kv_w;
    } else {
      qkv_w_ = q_w;
    }
    k_cache_ = k_cache;
    v_cache_ = v_cache;
    qkv_b_ = qkv_b;
    projection_w_ = projection_w;
    projection_b_ = projection_b;
    act_seq_arr_ = reinterpret_cast<int64_t *>(malloc(sizeof(int64_t) * p->batch_size_));
  }

  int Compute(void *input, void *mask, void *position_idx, void *embedding_table, void *q_seq, void *kv_seq,
              void *q_padding, void *kv_padding, void *mode, void *output, EncoderParams *p, void *ws, void *sys_ws,
              void *stream, void *alt_stream) {
    void *qkv_output = ws;
    void *query = reinterpret_cast<uint8_t *>(ws) + p->batch_size_ * p->seq_ * p->hid_dim_ * 3 * sizeof(T);
    void *ws_qkv = query;
    ComputeQKV(input, position_idx, embedding_table, q_seq, q_padding, mode, qkv_output, ws_qkv, sys_ws, p, stream,
               alt_stream);
    QKVPermuteAscendc(qkv_output, qkv_b_, query, k_cache_, v_cache_, q_seq, kv_seq, q_padding, kv_padding, mode,
                      p->token_num_, p->batch_size_, p->seq_, p->head_num_, p->head_size_, vec_cores_, stream);
    for (int i = 0; i < p->batch_size_; i++) {
      act_seq_arr_[i] = p->act_kv_seq_[i];
    }
    if (p->incremental_mode_ == false) {
      prepareAttentionExecuter(query, mask, p, &prompt_, ws);
      auto ret = aclnnPromptFlashAttention(sys_ws, prompt_.workspace_size_, prompt_.executor_, stream);
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "error in aclnnPromptFlashAttention " << ret;
        return ret;
      }
    } else {
      prepareAttentionExecuter(query, mask, p, &incremental_, ws);
      auto ret = aclnnIncreFlashAttention(sys_ws, incremental_.workspace_size_, incremental_.executor_, stream);
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "error in aclnnIncreFlashAttention " << ret << aclGetRecentErrMsg();
        return ret;
      }
    }
    // if batch equal to 1, no need to do squeeze
    void *proj_in = ws;
    if (p->batch_size_ > 1 && !p->incremental_mode_) {  // BSH -> TH (change to copy full H size)
      // do squeeze !
      size_t transpose_offset = p->batch_size_ * p->seq_ * p->hid_dim_ * sizeof(T);
      proj_in = reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(ws) + transpose_offset);
      Transpose0213Ascendc(ws, proj_in, q_seq, q_padding, mode, p->token_num_, p->batch_size_, p->seq_, p->head_num_,
                           p->head_size_, vec_cores_, stream);
    }
    gemm_projection_.execute(1, p->token_num_, p->hid_dim_, p->head_num_ * p->head_size_, proj_in, projection_w_,
                             output, sys_ws, SYS_WS_RESERVED, stream);
    if (p->rank_num_ > 1) {
      auto &hccl = HcclAdapter::GetInstance();
      uint64_t count = p->token_num_ * p->hid_dim_;
      hccl.AllSumReduce(output, output, count, stream);
    }
    return ACL_SUCCESS;
  }

  size_t GetWsSize(EncoderParams *p) { return 4 * p->batch_size_ * p->seq_ * p->hid_dim_ * sizeof(T); }

  ~FlashAttn() {
    DestroyExecuter();
    if (act_seq_arr_) free(act_seq_arr_);
    act_seq_arr_ = nullptr;
  }

 private:
  int ComputeQKV(void *input, void *position_idx, void *embedding_table, void *q_seq, void *q_padding, void *mode,
                 void *qkv_output, void *ws, void *sys_ws, EncoderParams *p, void *stream, void *alt_stream) {
    int actual_hid_size = p->head_num_ * p->head_size_;
    // void *query = ws;
    if (p->is_moe_ && !p->incremental_mode_) aclrtSynchronizeStream(alt_stream);
    if (p->is_query_) {
      void *act_stream = p->incremental_mode_ ? stream : alt_stream;
      // step I - multiply KV
      void *kv_out = reinterpret_cast<uint8_t *>(qkv_output) + actual_hid_size * sizeof(T);
      void *ws_kv = reinterpret_cast<uint8_t *>(sys_ws);
      gemm_kv_.execute(1, p->token_num_, 2 * actual_hid_size, p->hid_dim_, input, kv_w_, kv_out, ws_kv, SYS_WS_RESERVED,
                       stream, nullptr, p->hid_dim_, 2 * actual_hid_size, 3 * actual_hid_size);
      // step II - generate q signal from query embedding
      if (!fuse_embed_) {
        fuse_embed_ = true;
        Gemm gemm;
        void *temp = reinterpret_cast<uint8_t *>(sys_ws);
        int out_len = p->seq_ * actual_hid_size * sizeof(T);
        void *temp_ws = reinterpret_cast<uint8_t *>(sys_ws) + out_len;
        gemm.execute(1, p->seq_, actual_hid_size, p->hid_dim_, embedding_table, q_w_, temp, temp_ws, SYS_WS_RESERVED,
                     stream);
        aclrtMemcpyAsync(embedding_table, out_len, temp, out_len, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        aclrtSynchronizeStream(stream);
      }

      VocabEmbeddingAscendc(position_idx,
                            embedding_table,  // bias
                            qkv_output,       // query,
                            q_seq, q_padding, mode, p->token_num_, p->batch_size_, p->seq_, actual_hid_size, vec_cores_,
                            act_stream);
      // gemm_q_.execute(1, p->token_num_, actual_hid_size, p->hid_dim_, query, q_w_, qkv_output, sys_ws,
      //               SYS_WS_RESERVED / 2, act_stream, nullptr, p->hid_dim_, actual_hid_size, 3 * actual_hid_size);
      // step III - Make sure q sig is ready
      if (!p->incremental_mode_) aclrtSynchronizeStream(alt_stream);
    } else {
      gemm_qkv_.execute(1, p->token_num_, 3 * actual_hid_size, p->hid_dim_, input, qkv_w_, qkv_output, sys_ws,
                        SYS_WS_RESERVED, stream);
    }
    return ACL_SUCCESS;
  }

  void cleanupExecuter(AttnAclExecT *handle) {
    if (handle->q_) {
      aclDestroyTensor(handle->q_);
      handle->q_ = nullptr;
    }
    if (handle->k_list_) {
      aclDestroyTensorList(handle->k_list_);
      handle->k_list_ = nullptr;
      handle->k_ = nullptr;  // cleanup in list destroy
    } else {
      if (handle->k_) {
        aclDestroyTensor(handle->k_);
        handle->k_ = nullptr;
      }
    }
    if (handle->v_list_) {
      aclDestroyTensorList(handle->v_list_);
      handle->v_list_ = nullptr;
      handle->v_ = nullptr;  // cleanup in list destroy
    } else {
      if (handle->v_) {
        aclDestroyTensor(handle->v_);
        handle->v_ = nullptr;
      }
    }
    if (handle->mask_) {
      aclDestroyTensor(handle->mask_);
      handle->mask_ = nullptr;
    }
    if (handle->out_) {
      aclDestroyTensor(handle->out_);
      handle->out_ = nullptr;
    }
    if (handle->act_seq_) {
      aclDestroyIntArray(handle->act_seq_);
      handle->act_seq_ = nullptr;
    }
    handle->executor_ = nullptr;
  }

  int prepareAttentionExecuter(void *query, void *mask, EncoderParams *p, AttnAclExecT *handle, void *output) {
    cleanupExecuter(handle);
    bool is_inc = p->incremental_mode_;
    std::vector<int64_t> q_shape = {p->batch_size_, is_inc ? 1 : p->seq_, p->head_num_ * p->head_size_};
    auto q_stride = calcStride(q_shape);

    std::vector<int64_t> kv_shape = {p->batch_size_, p->seq_, p->head_num_ * p->head_size_};
    auto kv_stride = calcStride(kv_shape);

    std::vector<int64_t> mask_shape = {is_inc ? 1 : p->seq_, p->seq_};
    auto mask_stride = calcStride(mask_shape);

    handle->q_ = aclCreateTensor(q_shape.data(), q_shape.size(), aclDataType::ACL_FLOAT16, q_stride.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, q_shape.data(), q_shape.size(), query);

    handle->k_ = aclCreateTensor(kv_shape.data(), kv_shape.size(), aclDataType::ACL_FLOAT16, kv_stride.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, kv_shape.data(), kv_shape.size(), k_cache_);

    handle->v_ = aclCreateTensor(kv_shape.data(), kv_shape.size(), aclDataType::ACL_FLOAT16, kv_stride.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, kv_shape.data(), kv_shape.size(), v_cache_);

    handle->mask_ = aclCreateTensor(mask_shape.data(), mask_shape.size(), aclDataType::ACL_FLOAT16, mask_stride.data(),
                                    0, aclFormat::ACL_FORMAT_ND, mask_shape.data(), mask_shape.size(), mask);

    handle->out_ = aclCreateTensor(q_shape.data(), q_shape.size(), aclDataType::ACL_FLOAT16, q_stride.data(), 0,
                                   aclFormat::ACL_FORMAT_ND, q_shape.data(), q_shape.size(), output);

    handle->act_seq_ = aclCreateIntArray(act_seq_arr_, p->batch_size_);

    // B Batch, N head_num, D headDim, H head_size=head_num*headDim, S SequnceLength
    char format[] = "BSH";
    aclnnStatus ret = ACL_SUCCESS;
    if (is_inc) {
      handle->k_list_ = aclCreateTensorList(&handle->k_, 1);
      handle->v_list_ = aclCreateTensorList(&handle->v_, 1);
      ret = aclnnIncreFlashAttentionGetWorkspaceSize(handle->q_, handle->k_list_, handle->v_list_, nullptr, nullptr,
                                                     handle->act_seq_, p->head_num_, p->scale_, format, 0, handle->out_,
                                                     &handle->workspace_size_, &handle->executor_);
    } else {
      ret = aclnnPromptFlashAttentionGetWorkspaceSize(
        handle->q_, handle->k_, handle->v_, nullptr, handle->mask_, handle->act_seq_, p->head_num_, p->scale_,
        std::numeric_limits<int>::max(), 0, format, 0, handle->out_, &handle->workspace_size_, &handle->executor_);
    }
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "error in flush attention workspace size (" << ret << ") " << aclGetRecentErrMsg();
      return ret;
    }
    if (handle->workspace_size_ > SYS_WS_RESERVED) {
      MS_LOG(ERROR) << "error in flush attention workspace size too big:" << handle->workspace_size_;
      return -1;
    }
    return ACL_SUCCESS;
  }
  void DestroyExecuter() {
    cleanupExecuter(&prompt_);
    cleanupExecuter(&incremental_);
  }
  int cube_cores_;
  int vec_cores_;
  Gemm gemm_projection_;
  GemmDistrubute gemmd_projection_;
  Gemm gemm_q_;
  Gemm gemm_kv_;
  Gemm gemm_qkv_;

  void *qkv_w_ = nullptr;
  void *qkv_b_ = nullptr;
  void *q_w_ = nullptr;
  void *kv_w_ = nullptr;
  void *k_cache_ = nullptr;
  void *v_cache_ = nullptr;
  void *projection_w_ = nullptr;
  void *projection_b_ = nullptr;
  int64_t *act_seq_arr_ = nullptr;
  AttnAclExecT prompt_;
  AttnAclExecT incremental_;
  bool fuse_embed_ = false;
};
}  // namespace mindspore::ascend_native
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_FLASH_ATTENTION_H_
