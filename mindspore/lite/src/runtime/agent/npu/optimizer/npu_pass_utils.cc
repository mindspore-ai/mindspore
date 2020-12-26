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

#include "src/kernel_registry.h"
#include "src/ops/nhwc2nchw.h"
#include "src/ops/nchw2nhwc.h"
#include "src/runtime/agent/npu/optimizer/npu_pass_utils.h"
namespace mindspore::lite {
using kernel::KERNEL_ARCH::kCPU;
using kernel::KERNEL_ARCH::kNPU;
PrimitiveC *NPUPassUtils::CreateNchw2NhwcPrimitive() {
  flatbuffers::FlatBufferBuilder fbb(1024);
  auto val_offset = schema::CreateNchw2Nhwc(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, schema::PrimitiveType_Nchw2Nhwc, val_offset.o);
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
  auto *primitive = PrimitiveC::NewPrimitiveC<Nchw2Nhwc>(flatbuffers::GetRoot<schema::Primitive>(primitive_buf));
  free(primitive_buf);
  fbb.Clear();
  return primitive;
}

PrimitiveC *NPUPassUtils::CreateNhwc2NchwPrimitive() {
  flatbuffers::FlatBufferBuilder fbb(1024);
  auto val_offset = schema::CreateNhwc2Nchw(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, schema::PrimitiveType_Nhwc2Nchw, val_offset.o);
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
  auto *primitive = PrimitiveC::NewPrimitiveC<Nhwc2Nchw>(flatbuffers::GetRoot<schema::Primitive>(primitive_buf));
  free(primitive_buf);
  fbb.Clear();
  return primitive;
}

kernel::LiteKernel *NPUPassUtils::CreateNchw2NhwcKernel(const std::vector<Tensor *> &in_tensors,
                                                        const std::vector<Tensor *> &out_tensors,
                                                        const InnerContext *ctx, const std::string &name) {
  kernel::KernelKey key{kCPU, kNumberTypeFloat32, schema::PrimitiveType_Nchw2Nhwc};
  auto nchw2nhwc_primitive = CreateNchw2NhwcPrimitive();
  auto *nchw2nhwc_kernel =
    KernelRegistry::GetInstance()->GetKernel(in_tensors, out_tensors, nchw2nhwc_primitive, ctx, key);
  nchw2nhwc_kernel->set_name(name);
  return nchw2nhwc_kernel;
}

kernel::LiteKernel *NPUPassUtils::CreateNhwc2NchwKernel(const std::vector<Tensor *> &in_tensors,
                                                        const std::vector<Tensor *> &out_tensors,
                                                        const InnerContext *ctx, const std::string &name) {
  kernel::KernelKey key{kCPU, kNumberTypeFloat32, schema::PrimitiveType_Nhwc2Nchw};
  auto nhwc2nchw_primitive = CreateNhwc2NchwPrimitive();
  auto *nhwc2nchw_kernel =
    KernelRegistry::GetInstance()->GetKernel(in_tensors, out_tensors, nhwc2nchw_primitive, ctx, key);
  nhwc2nchw_kernel->set_name(name);
  return nhwc2nchw_kernel;
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
}  // namespace mindspore::lite
