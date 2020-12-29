/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/agent/npu/optimizer/npu_pass_utils.h"
#include "src/ops/transpose.h"
#include "nnacl/transpose.h"
#include "src/ops/populate/populate_register.h"
#include "src/runtime/kernel/arm/fp32/transpose_fp32.h"

namespace mindspore::lite {
using kernel::KERNEL_ARCH::kCPU;
using kernel::KERNEL_ARCH::kNPU;
PrimitiveC *NPUPassUtils::CreateTransposePrimitive() {
  flatbuffers::FlatBufferBuilder fbb(1024);
  auto val_offset = schema::CreateNchw2Nhwc(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, schema::PrimitiveType_Transpose, val_offset.o);
  fbb.Finish(prim_offset);
  auto buf = fbb.GetBufferPointer();
  if (buf == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer return nullptr";
    fbb.Clear();
    return nullptr;
  }
  auto primitive_buf = reinterpret_cast<char *>(malloc(fbb.GetSize()));
  if (primitive_buf == nullptr) {
    MS_LOG(ERROR) << "Malloc primitive buffer failed.";
    fbb.Clear();
    return nullptr;
  }
  memcpy(primitive_buf, buf, fbb.GetSize());
  auto *primitive = PrimitiveC::NewPrimitiveC<Transpose>(flatbuffers::GetRoot<schema::Primitive>(primitive_buf));
  free(primitive_buf);
  fbb.Clear();
  return primitive;
}

kernel::LiteKernel *NPUPassUtils::CreateNchw2NhwcKernel(const std::vector<Tensor *> &in_tensors,
                                                        const std::vector<Tensor *> &out_tensors,
                                                        const InnerContext *ctx, const std::string &name) {
  kernel::KernelKey key{kCPU, kNumberTypeFloat32, schema::PrimitiveType_Transpose};
  auto nchw2nhwc_primitive = CreateTransposePrimitive();
  auto *transpose_param = reinterpret_cast<TransposeParameter *>(malloc(sizeof(TransposeParameter)));
  if (transpose_param == nullptr) {
    MS_LOG(ERROR) << "malloc TransposeParameter failed.";
    return nullptr;
  }
  memset(transpose_param, 0, sizeof(TransposeParameter));
  transpose_param->op_parameter_.type_ = nchw2nhwc_primitive->Type();
  transpose_param->perm_[0] = 0;
  transpose_param->perm_[1] = 2;
  transpose_param->perm_[2] = 3;
  transpose_param->perm_[3] = 1;
  transpose_param->num_axes_ = 4;

  auto kernel = new (std::nothrow) kernel::TransposeCPUKernel(reinterpret_cast<OpParameter *>(transpose_param),
                                                              in_tensors, out_tensors, ctx, nchw2nhwc_primitive);
  if (kernel != nullptr) {
    kernel->set_desc(key);
  } else {
    MS_LOG(ERROR) << "New Nchw2Nhwc Kernel failed.";
    return nullptr;
  }

  kernel->set_name(name);
  return kernel;
}

kernel::LiteKernel *NPUPassUtils::CreateNhwc2NchwKernel(const std::vector<Tensor *> &in_tensors,
                                                        const std::vector<Tensor *> &out_tensors,
                                                        const InnerContext *ctx, const std::string &name) {
  kernel::KernelKey key{kCPU, kNumberTypeFloat32, schema::PrimitiveType_Transpose};
  auto nhwc2nchw_primitive = CreateTransposePrimitive();
  auto *transpose_param = reinterpret_cast<TransposeParameter *>(malloc(sizeof(TransposeParameter)));
  if (transpose_param == nullptr) {
    MS_LOG(ERROR) << "malloc TransposeParameter failed.";
    return nullptr;
  }
  memset(transpose_param, 0, sizeof(TransposeParameter));
  transpose_param->op_parameter_.type_ = nhwc2nchw_primitive->Type();
  transpose_param->perm_[0] = 0;
  transpose_param->perm_[1] = 3;
  transpose_param->perm_[2] = 1;
  transpose_param->perm_[3] = 2;
  transpose_param->num_axes_ = 4;

  auto kernel = new (std::nothrow) kernel::TransposeCPUKernel(reinterpret_cast<OpParameter *>(transpose_param),
                                                              in_tensors, out_tensors, ctx, nhwc2nchw_primitive);
  if (kernel != nullptr) {
    kernel->set_desc(key);
  } else {
    MS_LOG(ERROR) << "New Nhwc2Nchw Kernel failed.";
    return nullptr;
  }

  kernel->set_name(name);
  return kernel;
}

void NPUPassUtils::UpdateKernel(kernel::LiteKernel *kernel, const std::vector<kernel::LiteKernel *> &in_kernels,
                                const std::vector<kernel::LiteKernel *> &out_kernels,
                                const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors) {
  kernel->set_in_tensors(in_tensors);
  kernel->set_out_tensors(out_tensors);
  kernel->set_in_kernels(in_kernels);
  kernel->set_out_kernels(out_kernels);
}

void NPUPassUtils::UpdateNH2NCTransNodePreKernel(kernel::LiteKernel *pre_kernel, kernel::LiteKernel *trans_kernel,
                                                 kernel::LiteKernel *kernel) {
  std::vector<kernel::LiteKernel *> out_kernels;

  for (auto out_kernel : pre_kernel->out_kernels()) {
    if (out_kernel == kernel) {
      out_kernels.push_back(trans_kernel);
    } else {
      out_kernels.push_back(out_kernel);
    }
  }
  pre_kernel->set_out_kernels(out_kernels);
}

void NPUPassUtils::UpdateNC2NHTransNodePreKernel(kernel::LiteKernel *kernel, kernel::LiteKernel *trans_kernel,
                                                 kernel::LiteKernel *post_kernel) {
  std::vector<kernel::LiteKernel *> cur_out_kernels;
  for (auto out_kernel : kernel->out_kernels()) {
    if (out_kernel == post_kernel) {
      cur_out_kernels.push_back(trans_kernel);
    } else {
      cur_out_kernels.push_back(out_kernel);
    }
  }
  auto kernel_out_tensor = kernel->out_tensors()[0];
  // Change format the output of the current kernel nhwc->nchw
  auto nhwc_shape = kernel_out_tensor->shape();
  std::vector<int> nchw_shape = {nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]};
  kernel_out_tensor->set_format(schema::Format_NCHW);
  kernel_out_tensor->set_shape(nchw_shape);
  kernel->set_out_kernels(cur_out_kernels);
  kernel->set_out_tensors({kernel_out_tensor});
}

void NPUPassUtils::UpdateNH2NCTransNodeAfterKernel(kernel::LiteKernel *kernel, kernel::LiteKernel *trans_kernel,
                                                   kernel::LiteKernel *pre_kernel) {
  std::vector<lite::Tensor *> cur_kernel_in_tensors = {trans_kernel->out_tensors()[0]};
  for (int i = 1; i < kernel->in_tensors().size(); i++) {
    cur_kernel_in_tensors.push_back(kernel->in_tensors()[i]);
  }
  std::vector<kernel::LiteKernel *> cur_in_kernels = {trans_kernel};
  for (int i = 1; i < kernel->in_kernels().size(); i++) {
    auto in_kernel = kernel->in_kernels()[i];
    if (in_kernel != kernel) {
      cur_in_kernels.push_back(in_kernel);
    }
  }
  kernel->set_in_kernels(cur_in_kernels);
  kernel->set_in_tensors({cur_kernel_in_tensors});
}

void NPUPassUtils::UpdateNC2NHTransNodeAfterKernel(kernel::LiteKernel *kernel, kernel::LiteKernel *trans_kernel,
                                                   kernel::LiteKernel *post_kernel) {
  std::vector<Tensor *> post_in_tensors;
  for (auto post_in_tensor : post_kernel->in_tensors()) {
    if (post_in_tensor != kernel->out_tensors()[0]) {
      post_in_tensors.push_back(post_in_tensor);
    } else {
      post_in_tensors.push_back(trans_kernel->out_tensors()[0]);
    }
  }
  post_kernel->set_in_tensors(post_in_tensors);
  std::vector<kernel::LiteKernel *> post_in_kernels;
  for (auto in_kernel : post_kernel->in_kernels()) {
    if (in_kernel == kernel) {
      post_in_kernels.push_back(trans_kernel);
    } else {
      post_in_kernels.push_back(in_kernel);
    }
  }
  post_kernel->set_in_kernels(post_in_kernels);
  post_kernel->set_in_tensors({post_in_tensors});
}

bool NPUPassUtils::IsNhwc2Nchw(const kernel::LiteKernel *kernel) {
  if (kernel->Type() != schema::PrimitiveType_Transpose) {
    return false;
  }
  auto parameter = reinterpret_cast<TransposeParameter *>(kernel->op_parameter());
  if (parameter->num_axes_ != 4) {
    return false;
  }

  std::vector<int> perm = {parameter->perm_[0], parameter->perm_[1], parameter->perm_[2], parameter->perm_[3]};
  std::vector<int> nh2nc_perm = {0, 3, 1, 2};
  if (nh2nc_perm == perm) {
    return true;
  }
  return false;
}

bool NPUPassUtils::IsNchw2Nhwc(const kernel::LiteKernel *kernel) {
  if (kernel->Type() != schema::PrimitiveType_Transpose) {
    return false;
  }
  auto parameter = reinterpret_cast<TransposeParameter *>(kernel->op_parameter());
  if (parameter->num_axes_ != 4) {
    return false;
  }

  std::vector<int> perm = {parameter->perm_[0], parameter->perm_[1], parameter->perm_[2], parameter->perm_[3]};
  std::vector<int> nh2nc_perm = {0, 2, 3, 1};
  if (nh2nc_perm == perm) {
    return true;
  }
  return false;
}

}  // namespace mindspore::lite
