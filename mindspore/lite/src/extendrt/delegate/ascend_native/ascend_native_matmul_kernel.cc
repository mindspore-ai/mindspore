/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <vector>
#include "extendrt/delegate/ascend_native/ascend_native_matmul_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/ai_core/matmul.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "mindspore/lite/src/extendrt/delegate/ascend_native/ascend_native_impl/tiling.h"

#ifdef ACL_BLAS
#include <sstream>
#include <fstream>
#include "acl/ops/acl_cblas.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/copy_cast.h"
#endif
namespace mindspore::kernel {
using mindspore::ops::kNameMatMulFusion;

#ifdef ACL_BLAS

namespace {

std::string OmName(int m, int k, int n) {
  std::ostringstream os;
  os << "0_GEMM_";
  os << "1_2_" << m << "_" << k << "_";
  os << "1_2_" << k << "_" << n << "_";
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

int BuildFile(int m, int n, int k, bool ta, bool tb) {
  std::ostringstream os;
  os << "["
     << "\n";
  os << "{ "
     << "\n";
  os << "\"op\": \"GEMM\",";
  os << "\"input_desc\": [";
  os << BuildSection(m, k) << ",";
  os << BuildSection(k, n) << ",";
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
  std::ofstream fl("gemm.json");
  fl.write(os.str().c_str(), os.str().size());
  if (!fl.good()) {
    MS_LOG(ERROR) << "fail to write file" << os.str();
    return kError;
  }
  fl.close();
  return kSuccess;
}
}  // namespace
int AscendNativeMatmulKernel::PrepareBlas() {
  const char *pname = "op_models";
  aclFloat16 alpha = aclFloatToFloat16(1.0);
  aclFloat16 beta = aclFloatToFloat16(0.0);
  alpha_ = nullptr;
  beta_ = nullptr;
  ascend_native::CopyHostFp16ToDeviceFp16(&alpha, &alpha_, 1, const_cast<void *>(stream_),
                                          const_cast<void *>(acl_ctx_));
  ascend_native::CopyHostFp16ToDeviceFp16(&beta, &beta_, 1, const_cast<void *>(stream_), const_cast<void *>(acl_ctx_));
  std::string fullpath = std::string(pname) + "/" + OmName(m_, n_, k_);
  std::fstream f(fullpath.c_str());
  if (!f.good()) {
    // if file don't exits create json and compile
    auto res = BuildFile(m_, n_, k_, transpose_a_, transpose_b_);
    if (res != kSuccess) {
      MS_LOG(ERROR) << "could not create json file";
      return kError;
    }
    std::ostringstream os;
    os << "atc --singleop=gemm.json --soc_version=Ascend910B1 --log=debug --output=" << pname;
    res = system(os.str().c_str());
    if (res != 0) {
      MS_LOG(ERROR) << "running atc failed";
      return kError;
    }
  }
  aclopSetModelDir(pname);
  return kSuccess;
}
#endif

int AscendNativeMatmulKernel::InferShape() {
  if (in_tensors_[0] != nullptr && in_tensors_[1] != nullptr) {
    bool is_bmm = (in_tensors_[0]->shape().size() == C3NUM);
    std::vector<int> shape;
    if (is_bmm) shape.push_back(in_tensors_[0]->shape()[0]);
    shape.push_back(m_);
    shape.push_back(n_);
    out_tensors_[0]->set_shape(shape);
  }
  return kSuccess;
}

int AscendNativeMatmulKernel::Prepare() {
  auto primitive = AsOps<ops::MatMulFusion>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "convert to primitive matmul failed for " << get_name();
    return kLiteError;
  }
  transpose_a_ = primitive->get_transpose_a();
  transpose_b_ = primitive->get_transpose_b();
  bool is_bias = (in_tensors_.size() == C3NUM);
  bool is_bmm = (in_tensors_[0]->shape().size() == C3NUM);
  auto zeroth_mm_dim = in_tensors_[0]->shape().size() - 2;
  m_ = (transpose_a_) ? in_tensors_[0]->shape()[zeroth_mm_dim + C1NUM] : in_tensors_[0]->shape()[zeroth_mm_dim];
  k_ = (transpose_a_) ? in_tensors_[0]->shape()[zeroth_mm_dim] : in_tensors_[0]->shape()[zeroth_mm_dim + C1NUM];
  n_ = (transpose_b_) ? in_tensors_[C1NUM]->shape()[zeroth_mm_dim] : in_tensors_[C1NUM]->shape()[zeroth_mm_dim + C1NUM];
#ifdef ACL_BLAS
  if (!is_bmm) {
    PrepareBlas();
  } else {
#endif
    ascend_native::PrepareMatmul(&tile_data_d_, &tile_data_h_, m_, n_, k_, transpose_a_, transpose_b_, is_bias, is_bmm,
                                 const_cast<void *>(stream_), const_cast<void *>(acl_ctx_));
    if (is_bmm) {
      ascend_native::PrepareMatmulExtra(&extra_d_, &extra_h_, const_cast<void *>(stream_), const_cast<void *>(acl_ctx_),
                                        in_tensors_[0]->shape()[0]);
    }
#ifdef ACL_BLAS
  }
#endif
  return kSuccess;
}

int AscendNativeMatmulKernel::Run() {
  bool is_bmm = (extra_d_ && extra_h_.bmm_num_ > 1);
#ifdef ACL_BLAS
  if (!is_bmm) {
    auto ret = aclblasGemmEx(transpose_a_ == false ? ACL_TRANS_N : ACL_TRANS_T,
                             transpose_b_ == false ? ACL_TRANS_N : ACL_TRANS_T, ACL_TRANS_N, m_, n_, k_, alpha_,
                             in_tensors_[0]->device_data(), -1, ACL_FLOAT16, in_tensors_[1]->device_data(), -1,
                             ACL_FLOAT16, beta_, out_tensors_[0]->device_data(), -1, ACL_FLOAT16,
                             ACL_COMPUTE_HIGH_PRECISION, const_cast<void *>(stream_));
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "fails running aclblasGemmEx ret=" << ret;
      return kError;
    }
    return kSuccess;
  }
#endif
  ascend_native::MatmulAi(transpose_a_, transpose_b_, in_tensors_[0]->device_data(), in_tensors_[1]->device_data(),
                           out_tensors_[0]->device_data(), nullptr, tile_data_d_, tile_data_h_,
                           const_cast<void *>(get_sys_workspace()), is_bmm, extra_d_, const_cast<void *>(stream_),
                           const_cast<void *>(acl_ctx_));
  return kSuccess;
}

int AscendNativeMatmulKernel::ReSize() {
  if (in_tensors_[0]->shape()[1] != in_tensors_[1]->shape()[0]) {
    MS_LOG(ERROR) << "matmul ReSize failed";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
REGISTER_ASCEND_NATIVE_CREATOR(kNameMatMulFusion, AscendNativeMatmulKernel)
}  // namespace mindspore::kernel
