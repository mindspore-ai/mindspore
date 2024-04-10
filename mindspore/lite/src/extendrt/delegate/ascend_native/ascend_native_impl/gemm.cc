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

#include "extendrt/delegate/ascend_native/ascend_native_impl/gemm.h"
#include <sstream>
#include <fstream>
#include <string>
#include "src/common/log_adapter.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_addmm.h"
#include "aclnnop/aclnn_addbmm.h"
#include "acl/ops/acl_cblas.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/copy_cast.h"
#if MS_ENABLE_ASCEND_DISTRIBUTION
#include "extendrt/delegate/ascend_native/ascend_native_impl/hccl_adapter.h"
#endif

namespace mindspore::ascend_native {

namespace {

std::string OmName(int m, int n, int k, bool ta, bool tb) {
  std::ostringstream os;
  os << "0_GEMM_";
  if (!ta) {
    os << "1_2_" << m << "_" << k << "_";
  } else {
    os << "1_2_" << k << "_" << m << "_";
  }
  if (!tb) {
    os << "1_2_" << k << "_" << n << "_";
  } else {
    os << "1_2_" << n << "_" << k << "_";
  }
  os << "1_2_" << m << "_" << n << "_";
  os << "1_2_1_2_";
  os << "1_2_" << m << "_" << n << ".om";
  return os.str();
}

std::string BuildSection(int m, int n) {
  std::ostringstream os;
  os << "{" << std::endl;
  os << "\"format\": \"ND\"," << std::endl;
  os << "\"shape\": [" << m << "," << n << "]," << std::endl;
  os << "\"type\": \"float16\"" << std::endl;
  os << "}" << std::endl;
  return os.str();
}

std::string BuildEmptySection() {
  std::ostringstream os;
  os << "{" << std::endl;
  os << "\"format\": \"ND\"," << std::endl;
  os << "\"shape\": []," << std::endl;
  os << "\"type\": \"float16\"" << std::endl;
  os << "}" << std::endl;
  return os.str();
}

const std::string FileName() {
  const std::string fname = "gemm.json";
  return fname;
}

int BuildFile(int m, int n, int k, bool ta, bool tb) {
  std::ostringstream os;
  os << "["
     << "\n";
  os << "{ "
     << "\n";
  os << "\"op\": \"GEMM\",";
  os << "\"input_desc\": [";
  if (!ta) {
    os << BuildSection(m, k) << ",";
  } else {
    os << BuildSection(k, m) << ",";
  }
  if (!tb) {
    os << BuildSection(k, n) << ",";
  } else {
    os << BuildSection(n, k) << ",";
  }
  os << BuildSection(m, n) << ",";
  os << BuildEmptySection() << ",";
  os << BuildEmptySection() << "],";
  os << "\"output_desc\": [";
  os << BuildSection(m, n);
  os << "],";
  os << "\"attr\": [";
  os << "{ ";
  os << "\"name\": \"transpose_a\",";
  os << "\"type\": \"bool\",";
  os << "\"value\":" << (ta ? "true" : "false");
  os << "},";
  os << "{";
  os << "\"name\": \"transpose_b\",";
  os << "\"type\": \"bool\",";
  os << "\"value\":" << (tb ? "true" : "false");
  os << "}]}]";

  auto fname = FileName();
  std::ofstream fl(fname.c_str());
  fl.write(os.str().c_str(), os.str().size());
  if (!fl.good()) {
    MS_LOG(ERROR) << "fail to write file" << os.str();
    return ACL_ERROR_WRITE_FILE;
  }
  fl.close();
  return ACL_SUCCESS;
}
}  // namespace

aclnnStatus Gemm::init(int B, int M, int N, int K, void *a, void *b, void *c, void *stream, bool ta, bool tb,
                       void *bias, int lda, int ldb, int ldc) {
  is_blas_ = false;
  if ((ta || tb) && (B == 1) && (lda == -1) & (ldb == -1) && (ldc == -1) && (bias == nullptr)) {
    auto res = initAclBlas(M, N, K, a, b, c, ta, tb, stream);
    if (res != ACL_SUCCESS) {
      MS_LOG(ERROR) << "failed to init blas";
      return res;
    }
    is_blas_ = true;
  } else if (!ta && !tb) {
    auto res = initAclNN(B, M, N, K, a, b, c, bias, lda, ldb, ldc);
    if (res != ACL_SUCCESS) {
      MS_LOG(ERROR) << "failed to init aclnn";
      return res;
    }
  } else {
    MS_LOG(ERROR) << " not supported";
    return ACL_ERROR_INVALID_PARAM;
  }
  return ACL_SUCCESS;
}

aclnnStatus Gemm::initAclBlas(int M, int N, int K, void *a, void *b, void *c, bool ta, bool tb, void *stream) {
  const char *pname = "op_models";
  aclFloat16 alpha = aclFloatToFloat16(1.0);
  aclFloat16 beta = aclFloatToFloat16(0.0);
  alpha_ = nullptr;
  beta_ = nullptr;
  ascend_native::CopyHostFp16ToDeviceFp16(&alpha, &alpha_, 1, const_cast<void *>(stream));
  ascend_native::CopyHostFp16ToDeviceFp16(&beta, &beta_, 1, const_cast<void *>(stream));
  std::string fullpath = std::string(pname) + "/" + OmName(M, N, K, ta, tb);
  std::fstream f(fullpath.c_str());
  if (!f.good()) {
#ifdef MS_ENABLE_ASCEND_DISTRIBUTION
    auto &hccl = HcclAdapter::GetInstance();
    rank_ = hccl.get_rank();
#endif
    if (rank_ == 0) {
      // if file don't exits create json and compile
      auto res = BuildFile(M, N, K, ta, tb);
      if (res != ACL_SUCCESS) {
        MS_LOG(ERROR) << "could not create json file";
        return res;
      }
      std::ostringstream os;
      auto &fname = FileName();
      os << "atc --singleop=" << fname << " --soc_version=Ascend910B1 --log=debug --output=" << pname;
      res = system(os.str().c_str());
      if (res != 0) {
        MS_LOG(ERROR) << "running atc failed";
        return res;
      }
    }
#ifdef MS_ENABLE_ASCEND_DISTRIBUTION
    hccl.Sync();
#endif
  }
  aclopSetModelDir(pname);
  matrix_a_ = a;
  matrix_b_ = b;
  matrix_c_ = c;
  transpose_a_ = ta;
  transpose_b_ = tb;
  m_ = M;
  n_ = N;
  k_ = K;
  return ACL_SUCCESS;
}

aclnnStatus Gemm::initAclNN(int B, int M, int N, int K, void *a, void *b, void *c, void *bias, int lda, int ldb,
                            int ldc) {
  std::vector<int64_t> shapeA, shapeB, shapeC;
  std::vector<int64_t> shapeAv, shapeBv, shapeCv;
  clean();
  if (lda == -1) {
    lda = K;
  }
  if (ldb == -1) {
    ldb = N;
  }
  if (ldc == -1) {
    ldc = N;
  }
  if (B == 1) {
    shapeA = {M, K};
    shapeB = {K, N};
    shapeC = {M, N};
    shapeAv = {M, lda};
    shapeBv = {K, ldb};
    shapeCv = {M, ldc};
  } else {
    shapeA = {B, M, K};
    shapeB = {B, K, N};
    shapeC = {B, M, N};
    shapeAv = {B, M, lda};
    shapeBv = {B, K, ldb};
    shapeCv = {B, M, ldc};
  }
  auto strideA = calcStride(shapeAv);
  auto strideB = calcStride(shapeBv);
  auto strideC = calcStride(shapeCv);
  auto tensorA = aclCreateTensor(shapeA.data(), shapeA.size(), aclDataType::ACL_FLOAT16, strideA.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, shapeAv.data(), shapeAv.size(), a);
  auto tensorB = aclCreateTensor(shapeB.data(), shapeB.size(), aclDataType::ACL_FLOAT16, strideB.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, shapeBv.data(), shapeBv.size(), b);
  auto tensorC = aclCreateTensor(shapeC.data(), shapeC.size(), aclDataType::ACL_FLOAT16, strideC.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, shapeCv.data(), shapeCv.size(), c);

  vcollect_ = {tensorA, tensorB, tensorC};
  aclnnStatus ret;
  is_bias_ = false;
  is_bmm_ = false;
  if (bias != nullptr) {
    is_bias_ = true;
    std::vector<int64_t> shapeBias = {N};
    auto strideBias = calcStride(shapeBias);
    auto tensorBias = aclCreateTensor(shapeBias.data(), shapeBias.size(), aclDataType::ACL_FLOAT16, strideBias.data(),
                                      0, aclFormat::ACL_FORMAT_ND, shapeBias.data(), shapeBias.size(), bias);
    vcollect_.push_back(tensorBias);
    float alphaValue = 1.0f;
    float betaValue = 1.0f;
    auto alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    auto beta = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
    scollect_ = {alpha, beta};
    if (B > 1) {
      is_bmm_ = true;
      ret = aclnnAddbmmGetWorkspaceSize(tensorBias, tensorA, tensorB, beta, alpha, tensorC, 0, &workspace_size_,
                                        &executor_);
    } else {
      ret =
        aclnnAddmmGetWorkspaceSize(tensorBias, tensorA, tensorB, beta, alpha, tensorC, 0, &workspace_size_, &executor_);
    }

  } else {
    ret = aclnnMatmulGetWorkspaceSize(tensorA, tensorB, tensorC, 0, &workspace_size_, &executor_);
  }
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "error in maxmul workspace size " << ret;
    return ret;
  }
  return ACL_SUCCESS;
}
aclnnStatus Gemm::compute(void *workspace, uint64_t workspace_size, void *stream) {
  aclnnStatus ret;
  if (is_blas_) {
    MS_LOG(INFO) << "M=" << m_ << " N=" << n_ << " K=" << k_ << " ta=" << transpose_a_ << " tb=" << transpose_b_;
    ret = aclblasGemmEx(transpose_a_ == false ? ACL_TRANS_N : ACL_TRANS_T,
                        transpose_b_ == false ? ACL_TRANS_N : ACL_TRANS_T, ACL_TRANS_N, m_, n_, k_, alpha_, matrix_a_,
                        -1, ACL_FLOAT16, matrix_b_, -1, ACL_FLOAT16, beta_, matrix_c_, -1, ACL_FLOAT16,
                        ACL_COMPUTE_HIGH_PRECISION, stream);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "fails running aclblasGemmEx ret=" << ret << "aclGetRecentErrMsg() = " << aclGetRecentErrMsg();
      return ret;
    }
    return ACL_SUCCESS;
  }

  if (workspace_size < workspace_size_) {
    MS_LOG(ERROR) << "workspace is insufficient " << workspace_size << "< " << workspace_size_;
    return ACL_ERROR_STORAGE_OVER_LIMIT;
  }
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "no executor - init not completed";
    return ACL_ERROR_FAILURE;
  }

  if (is_bias_) {
    if (is_bmm_) {
      ret = aclnnAddbmm(workspace, workspace_size_, executor_, stream);
    } else {
      ret = aclnnAddmm(workspace, workspace_size_, executor_, stream);
    }
  } else {
    ret = aclnnMatmul(workspace, workspace_size_, executor_, stream);
  }
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "error in mat mul deploy (" << ret << ") " << aclGetRecentErrMsg();
    return ret;
  }
  return ACL_SUCCESS;
}
aclnnStatus Gemm::execute(int B, int M, int N, int K, void *a, void *b, void *c, void *workspacePtr,
                          uint64_t workspace_size, void *stream, void *bias, int lda, int ldb, int ldc) {
  auto ret = initAclNN(B, M, N, K, a, b, c, bias, lda, ldb, ldc);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "fail to init matmul " << ret;
    return ret;
  }
  ret = compute(workspacePtr, workspace_size, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "fail to compute matmul M=" << M << " N=" << N << " K=" << K << " ret=" << ret;
    clean();
    return ret;
  }
  return ACL_SUCCESS;
}

std::vector<int64_t> Gemm::calcStride(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

void Gemm::clean() {
  executor_ = nullptr;
  for (auto &item : vcollect_) {
    aclDestroyTensor(item);
  }
  vcollect_.clear();
  for (auto &item : scollect_) {
    aclDestroyScalar(item);
  }
  scollect_.clear();
}

aclnnStatus GemmDistrubute::execute(int M, int N, int K, int div, void *a, void *b, void *c, void *bias,
                                    void *workspace, uint64_t workspace_size, void *alt_stream, void *stream) {
  auto offset = [](void *ptr, int num) { return static_cast<void *>(static_cast<aclFloat16 *>(ptr) + num); };
  Gemm gemm;
  auto &hccl = HcclAdapter::GetInstance();
  int residual = M % div;
  int m = M / div;
  if (m > 0) {
    aclrtSynchronizeStream(stream);
    for (int i = 0; i < div; i++) {
      gemm.execute(1, m, N, K, offset(a, i * m * K), b, offset(c, i * m * N), workspace, workspace_size, alt_stream,
                   bias);
      aclrtSynchronizeStream(alt_stream);
      hccl.AllSumReduce(offset(c, i * m * N), offset(c, i * m * N), m * N, stream);
    }
  }
  if (residual > 0) {
    gemm.execute(1, residual, N, K, offset(a, div * m * K), b, offset(c, div * m * N), workspace, workspace_size,
                 stream, bias);
    hccl.AllSumReduce(offset(c, div * m * N), offset(c, div * m * N), residual * N, stream);
  }
  return ACL_SUCCESS;
}

}  // namespace mindspore::ascend_native
