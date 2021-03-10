/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/akg/gpu/akg_gpu_kernel_build.h"
#include <Python.h>
#include <vector>
#include <memory>
#include <string>
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/akg/gpu/akg_gpu_kernel_mod.h"
#include "utils/ms_utils.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
constexpr int32_t ARGS_SIZE = 1;
constexpr auto kCompileWithJsonFunc = "compilewithjson";

KernelPackPtr AkgGpuKernelBuilder::AkgSearchCache(const std::string &kernel_name, const std::string &processor) {
  return SearchCache(kernel_name, processor);
}

KernelPackPtr AkgGpuKernelBuilder::AkgInsertCache(const std::string &kernel_name, const std::string &processor) {
  return InsertCache(kernel_name, processor);
}

void AkgGpuKernelBuilder::AkgSetKernelMod(const KernelPackPtr &kernel_pack,
                                          const AkgKernelJsonGenerator &json_generator, const AnfNodePtr &anf_node) {
  auto kernel_mod_ptr = std::make_shared<GpuKernelMod>(kernel_pack);
  kernel_mod_ptr->SetInputSizeList(json_generator.input_size_list());
  kernel_mod_ptr->SetOutputSizeList(json_generator.output_size_list());
  AnfAlgo::SetKernelMod(kernel_mod_ptr, anf_node.get());
}

void AkgGpuKernelBuilder::AkgSaveJsonInfo(const string &kernel_name, const string &kernel_json) {
  kernel::SaveJsonInfo(kernel_name, kernel_json, kernel::KernelMeta::GetInstance()->kernel_meta_path());
}

KernelPackPtr AkgGpuKernelBuilder::OpBuild(const AkgKernelJsonGenerator &json_generator, const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto processor = GetProcessorStr(anf_node);
  auto kernel_name = json_generator.kernel_name();
  auto cached_kernel_pack = SearchCache(kernel_name, processor);
  if (cached_kernel_pack != nullptr) {
    MS_LOG(INFO) << "Use cached kernel, kernel_name[" << kernel_name << "], fullname_with_scope["
                 << anf_node->fullname_with_scope() << "].";
    return cached_kernel_pack;
  }

  auto kernel_json = json_generator.kernel_json_str();
  kernel::SaveJsonInfo(kernel_name, kernel_json, kernel::KernelMeta::GetInstance()->kernel_meta_path());
  (void)alarm(AUTODIFF_COMPILE_OVERTIME);
  auto res = GpuKernelBuildClient::Instance().AkgCompileSingle(kernel_json);
  (void)alarm(0);
  if (!res) {
    MS_LOG(ERROR) << "Akg compile failed, json: " << kernel_json;
    return nullptr;
  }

  auto new_kernel_pack = InsertCache(kernel_name, processor);
  if (new_kernel_pack == nullptr) {
    MS_LOG(ERROR) << "Insert to cache failed, kernel_name[" << kernel_name << "], fullname_with_scope["
                  << anf_node->fullname_with_scope() << "].";
    return nullptr;
  }
  return new_kernel_pack;
}

KernelModPtr AkgGpuKernelBuilder::BuildByJson(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_LOG(INFO) << "Akg start compile, op[" << anf_node->fullname_with_scope() << "]";
  AkgKernelJsonGenerator json_generator;
  if (!json_generator.CollectJson(anf_node)) {
    MS_LOG(ERROR) << "Op[" << anf_node->fullname_with_scope() << "] create single kernel json failed.";
  }

  auto kernel_pack = OpBuild(json_generator, anf_node);
  if (kernel_pack == nullptr) {
    MS_LOG(ERROR) << "Akg build failed op[" << anf_node->fullname_with_scope() << "].";
    return nullptr;
  }

  auto kernel_mod_ptr = std::make_shared<GpuKernelMod>(kernel_pack);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  kernel_mod_ptr->SetInputSizeList(json_generator.input_size_list());
  kernel_mod_ptr->SetOutputSizeList(json_generator.output_size_list());
  MS_LOG(INFO) << "Akg compile success, op[" << anf_node->fullname_with_scope() << "]";
  return kernel_mod_ptr;
}

KernelModPtr AkgGpuKernelBuilder::FuseByJson(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_LOG(INFO) << "Akg start compile, graph_kernel[" << anf_node->fullname_with_scope() << "]";
  auto fg = AnfAlgo::GetCNodeFuncGraphPtr(anf_node);
  MS_EXCEPTION_IF_NULL(fg);
  auto mng = fg->manager();
  if (mng == nullptr) {
    mng = Manage(fg, true);
    fg->set_manager(mng);
  }

  AnfNodePtrList node_list;
  AnfNodePtrList input_list;
  AnfNodePtrList output_list;
  GetValidKernelNodes(fg, &node_list, &input_list, &output_list);
  AkgKernelJsonGenerator json_generator;
  if (!json_generator.CollectFusedJson(node_list, input_list, output_list)) {
    MS_LOG(ERROR) << "Op[" << anf_node->fullname_with_scope() << "] create single kernel json failed.";
  }

  auto kernel_pack = OpBuild(json_generator, anf_node);
  if (kernel_pack == nullptr) {
    MS_LOG(ERROR) << "Akg build failed, graph_kernel[" << anf_node->fullname_with_scope() << "].";
    return nullptr;
  }

  auto kernel_mod_ptr = std::make_shared<GpuKernelMod>(kernel_pack);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  kernel_mod_ptr->SetInputSizeList(json_generator.input_size_list());
  kernel_mod_ptr->SetOutputSizeList(json_generator.output_size_list());
  MS_LOG(INFO) << "Akg compile success, graph_kernel[" << anf_node->fullname_with_scope() << "]";
  return kernel_mod_ptr;
}

KernelModPtr AkgGpuKernelBuild(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  AkgGpuKernelBuilder akg_gpu_kernel_builder;
  if (AnfAlgo::IsGraphKernel(anf_node)) {
    return akg_gpu_kernel_builder.FuseByJson(anf_node);
  }
  return akg_gpu_kernel_builder.BuildByJson(anf_node);
}
}  // namespace kernel
}  // namespace mindspore
