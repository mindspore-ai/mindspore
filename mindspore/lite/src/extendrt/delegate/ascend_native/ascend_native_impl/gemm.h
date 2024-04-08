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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_GEMM_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_GEMM_H_

#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
namespace mindspore::ascend_native {
class Gemm {
 public:
  virtual ~Gemm() { clean(); }
  Gemm() = default;
  aclnnStatus init(int B, int M, int N, int K, void *a, void *b, void *c, void *stream, bool ta = false,
                   bool tb = false, void *bias = nullptr, int lda = -1, int ldb = -1, int ldc = -1);
  aclnnStatus compute(void *workspace, uint64_t workspace_size, void *stream);
  aclnnStatus execute(int B, int M, int N, int K, void *a, void *b, void *c, void *workspacePtr,
                      uint64_t workspace_size, void *stream, void *bias = nullptr, int lda = -1, int ldb = -1,
                      int ldc = -1);

 private:
  std::vector<int64_t> calcStride(const std::vector<int64_t> &shape);
  aclnnStatus initAclNN(int B, int M, int N, int K, void *a, void *b, void *c, void *bias = nullptr, int lda = -1,
                        int ldb = -1, int ldc = -1);
  aclnnStatus initAclBlas(int M, int N, int K, void *a, void *b, void *c, bool ta, bool tb, void *stream);
  void clean();
  uint64_t workspace_size_ = 0;
  aclOpExecutor *executor_ = nullptr;
  std::vector<aclTensor *> vcollect_;
  std::vector<aclScalar *> scollect_;
  bool is_bias_ = false;
  bool is_bmm_ = false;
  bool is_blas_ = false;
  bool transpose_a_ = false;
  bool transpose_b_ = false;
  void *matrix_a_ = nullptr;
  void *matrix_b_ = nullptr;
  void *matrix_c_ = nullptr;
  int m_;
  int n_;
  int k_;
  void *alpha_;
  void *beta_;
  int rank_{0};
};

class GemmDistrubute {
 public:
  aclnnStatus execute(int M, int N, int K, int div, void *a, void *b, void *c, void *bias, void *workspace,
                      uint64_t workspace_size, void *alt_stream, void *stream);
};

};  // namespace mindspore::ascend_native
#endif
