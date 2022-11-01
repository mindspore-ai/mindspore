/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/cdist_grad_cpu_kernel.h"
#include <utility>
#include <algorithm>
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCdistInputDimsMin = 2;

const std::vector<KernelAttr> kernel_attr = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}};
}  // namespace

inline float DistSign(float val) {
  return std::min(std::max(0.f, std::ceil(val)), (1.f)) + std::min(std::max((-1.f), std::floor(val)), (0.f));
}

float CdistOneNormalcompute(float diff, float grad, float dist, float p) { return grad * DistSign(diff); }

float CdistLessTwoNormalcompute(float diff, float grad, float dist, float p) {
  if (diff == 0.0 || p < 1.0) {
    return 0.f;
  }
  return (DistSign(diff) * std::pow(std::abs(diff), (p - 1)) * grad / std::pow(dist, (p - 1)));
}

float CdistTwoNormalcompute(float diff, float grad, float dist, float p) {
  return dist == 0.0 ? 0.f : grad * diff / dist;
}

float CdistInfNormalcompute(float diff, float grad, float dist, float p) {
  return grad * DistSign(diff) * (1 - std::min(1.f, std::ceil(std::abs(std::abs(diff) - dist))));
}

float CdistPNormalcompute(float diff, float grad, float dist, float p) {
  float result;

  if (dist == 0.0) {
    result = 0.f;
  } else {
    result = diff * std::pow(std::abs(diff), (p - 2)) * grad / std::pow(dist, (p - 1));
  }
  return result;
}

void CdistGradCpuKernelMod::InitFunc(float p) {
  if (p == 0.0) {
    dist_func_ = nullptr;
  } else if (p == 1.0) {
    dist_func_ = CdistOneNormalcompute;
  } else if (p < 2.0) {
    dist_func_ = CdistLessTwoNormalcompute;
  } else if (p == 2.0) {
    dist_func_ = CdistTwoNormalcompute;
  } else if (std::isinf(p)) {
    dist_func_ = CdistInfNormalcompute;
  } else {
    dist_func_ = CdistPNormalcompute;
  }
}

bool CdistGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::CdistGrad>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "cast Cdist grad ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  p_ = kernel_ptr->get_p();

  auto input_type_id = inputs[0]->GetDtype();
  switch (input_type_id) {
    case kNumberTypeFloat32:
      InitFunc(p_);
      break;
    default:
      MS_LOG(ERROR) << "cdist grad kernel does not support " << TypeIdToString(input_type_id);
      return false;
  }
  return true;
}

int CdistGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &others) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, others);
  if (ret != 0) {
    MS_LOG(WARNING) << "For " << kernel_name_ << " Resize failed. ret " << ret;
    return ret;
  }
  std::vector<int64_t> in_shape0 = inputs[1]->GetShapeVector();
  std::vector<int64_t> in_shape1 = inputs[2]->GetShapeVector();
  auto in_shape_size = in_shape0.size();
  if (in_shape1.size() != in_shape_size || in_shape_size < kCdistInputDimsMin) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ",invalid input shape, input0 shape size " << in_shape_size
                  << ", input1 shape size " << in_shape1.size();
    return KRET_RESIZE_FAILED;
  }
  batch_ = 0;
  for (size_t i = 0; i < in_shape_size - kCdistInputDimsMin; i++) {
    batch_ += in_shape0[i];
  }
  batch_ = (batch_ <= 0) ? 1 : batch_;

  r0_ = in_shape0[in_shape_size - 2];
  m_ = in_shape0[in_shape_size - 1];
  r1_ = in_shape1[in_shape_size - 2];

  l1_size = r0_ * m_;
  l2_size = r1_ * m_;

  return 0;
}

std::vector<KernelAttr> CdistGradCpuKernelMod::GetOpSupport() { return kernel_attr; }

bool CdistGradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs) {
  float *grad_start = reinterpret_cast<float *>(inputs[0]->addr);
  float *dist_start = reinterpret_cast<float *>(inputs[3]->addr);
  float *t1_start = reinterpret_cast<float *>(inputs[1]->addr);
  float *t2_start = reinterpret_cast<float *>(inputs[2]->addr);
  float *res_start = reinterpret_cast<float *>(outputs[0]->addr);
  auto ret = memset_s(res_start, outputs[0]->size, 0, batch_ * r0_ * r1_ * sizeof(float));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << ret;
  }
  if (p_ == 0.0) {
    return true;
  }

  auto task = [this, grad_start, dist_start, t1_start, t2_start, res_start](size_t b_start, size_t b_end) {
    const float *i = t1_start;
    const float *j = t2_start;
    float *res_end = res_start + b_end;
    for (float *res_l = res_start + b_start; res_l != res_end; i += 1, j += 1, res_l += 1) {
      const float *t1 = i;
      const float *t2 = j;
      float *res = res_l;
      const float *t1_end = t1 + l1_size;
      const float *t2_end = t2 + l2_size;
      auto grad_k = grad_start;
      auto dist_k = dist_start;

      for (int64_t l = 0; l < batch_; l++) {
        for (; t1 != t1_end; t1 += m_, res += m_) {
          float t1_tmp = *t1;
          float res_tmp = *res;

          for (const float *t2_curr = t2; t2_curr != t2_end; t2_curr += m_, grad_k += 1, dist_k += 1) {
            auto diff = t1_tmp - *t2_curr;
            float res_curr = dist_func_(diff, (*grad_k), (*dist_k), p_);
            res_tmp = res_tmp + res_curr;
          }

          *res = res_tmp;
        }
        t1_end += l1_size;
        t2_end += l2_size;
        t2 += l2_size;
      }
    }
  };
  ParallelLaunchAutoSearch(task, m_, this, &parallel_search_info_, pool_);

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CdistGrad, CdistGradCpuKernelMod);

};  // namespace kernel

}  // namespace mindspore
