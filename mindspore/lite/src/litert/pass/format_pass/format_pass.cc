/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/pass/format_pass/format_pass.h"
#include "src/litert/pass/format_pass/insert_transpose.h"
#include "src/litert/pass/format_pass/eliminate_transpose.h"
#ifdef ENABLE_MULTI_LAYOUT
#include "src/litert/kernel_registry.h"
#include "nnacl/format_transpose_parameter.h"
#endif
#include "src/common/draw/drawer.h"

namespace mindspore::lite::pass {
#ifdef ENABLE_MULTI_LAYOUT
namespace {
kernel::KernelExec *DefaultCreateFormatTransFunc(Tensor *input, Tensor *output, const TransInfoPair &trans_info,
                                                 const std::string &name, const lite::InnerContext *ctx,
                                                 const kernel::KernelKey &desc) {
  auto param = reinterpret_cast<FormatTransposeParameter *>(malloc(sizeof(FormatTransposeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Malloc FormatTransposeParameter failed.";
    return nullptr;
  }
  (void)memset(param, 0, sizeof(FormatTransposeParameter));
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_FormatTranspose);
  param->src_format_ = static_cast<FormatC>((trans_info.src_format_));
  param->dst_format_ = static_cast<FormatC>((trans_info.dst_format_));
  kernel::KernelKey format_transpose_key = desc;
  format_transpose_key.type = schema::PrimitiveType_FormatTranspose;
  format_transpose_key.format = NHWC;
  format_transpose_key.data_type = input->data_type();

  kernel::MSKernel *kernel_impl;
  auto lite_kernel = KernelRegistry::GetInstance()->GetLiteKernel({input}, {output}, ctx, &format_transpose_key,
                                                                  reinterpret_cast<OpParameter *>(param));
  if (lite_kernel == nullptr) {
    MS_LOG(ERROR) << "Create format-transpose lite-kernel failed.";
    free(param);
    return nullptr;
  }
  kernel_impl = lite_kernel;
  auto *kernel_exec = new (std::nothrow) kernel::KernelExec(std::shared_ptr<kernel::MSKernel>(kernel_impl));
  if (kernel_exec == nullptr) {
    MS_LOG(ERROR) << "Create format-transpose kernel-exec failed.";
    return nullptr;
  }
  kernel_exec->set_desc(format_transpose_key);
  kernel_exec->set_context(ctx);
  kernel_exec->set_name(name);
  return kernel_exec;
}
}  // namespace
#endif

int FormatOptimize::AddPass(const FormatPassPtr &pass) {
  CHECK_NULL_RETURN(pass);
  pass_list_.push_back(pass);
  return RET_OK;
}

int FormatOptimize::RunPass(kernel::SubGraphKernel *graph, std::vector<Tensor *> *tensors) {
  for (const FormatPassPtr &pass : pass_list_) {
    CHECK_NULL_RETURN(pass);

    auto status = pass->RunPass(graph, tensors);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run pass failed";
      return status;
    }
    DrawDot(graph, pass->name());
  }
  return RET_OK;
}

int DoFormatPass(std::vector<mindspore::kernel::KernelExec *> *subgraph_list,
                 std::vector<mindspore::lite::Tensor *> *tensors, mindspore::Format graph_format,
                 const CreateFormatTransposeFunc &create_format_transpose_func) {
  for (const auto &subgraph : *subgraph_list) {
    FormatOptimizePtr optimize = std::make_shared<FormatOptimize>();

    (void)optimize->AddPass(std::make_shared<InsertTranspose>(graph_format, create_format_transpose_func));
    (void)optimize->AddPass(std::make_shared<EliminateTranspose>(graph_format, create_format_transpose_func));

    auto graph = reinterpret_cast<kernel::SubGraphKernel *>(subgraph);
    auto ret = optimize->RunPass(graph, tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Runtime format pass failed.";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int RuntimeFormatPass(std::vector<mindspore::kernel::KernelExec *> *subgraph_list,
                      std::vector<mindspore::lite::Tensor *> *tensors, mindspore::Format graph_format,
                      const CreateFormatTransposeFunc &create_format_transpose_func) {
#ifndef ENABLE_MULTI_LAYOUT
  return RET_OK;
#else
  if (create_format_transpose_func == nullptr) {
    return DoFormatPass(subgraph_list, tensors, graph_format, DefaultCreateFormatTransFunc);
  } else {
    return DoFormatPass(subgraph_list, tensors, graph_format, create_format_transpose_func);
  }
#endif
}
}  // namespace mindspore::lite::pass
