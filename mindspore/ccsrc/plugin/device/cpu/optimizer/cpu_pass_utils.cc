/**
 * Copyright  2019-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/optimizer/cpu_pass_utils.h"
#include <memory>
#include <string>

#include "ops/array_ops.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/backend/optimizer/helper.h"
#include "kernel/kernel_build_info.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_graph.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#include "kernel/framework_utils.h"

namespace mindspore {
namespace opt {
AnfNodePtr AddCastOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const std::string &format,
                                const TypeId &input_type, const TypeId &output_type,
                                const abstract::BaseShapePtr &origin_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(origin_shape);
  std::string input_format = format;
  std::string output_format = format;
  CNodePtr cast = func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())), input});
  MS_EXCEPTION_IF_NULL(cast);
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({input_format});
  builder.SetOutputsFormat({output_format});
  builder.SetInputsDeviceType({input_type});
  builder.SetOutputsDeviceType({output_type});
  if (cast->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    cast->set_kernel_info(kernel_info);
  }
  if (origin_shape->IsDynamic()) {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), cast);
    common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), cast);
  }
  common::AnfAlgo::SetNodeAttr(kAttrDstType, TypeIdToType(output_type), cast);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cast.get());
  common::AnfAlgo::SetOutputTypeAndDetailShape({output_type}, {origin_shape}, cast.get());
  common::AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(true), cast);
  std::shared_ptr<kernel::NativeCpuKernelMod> cpu_kernel =
    kernel::Factory<kernel::NativeCpuKernelMod>::Instance().Create(kCastOpName);
  if (cpu_kernel == nullptr) {
    MS_LOG(EXCEPTION) << "Operator[Cast] " << cast->kernel_info() << " is not support.";
  }

  auto kernel_attrs = cpu_kernel->GetOpSupport();
  kernel::SetCpuRefMapToKernelInfo(cast, kernel_attrs);
  auto thread_pool = kernel::GetActorMgrInnerThreadPool();
  cpu_kernel->SetThreadPool(thread_pool);
  auto args = kernel::AbstractArgsFromCNode(cast);
  auto op = kernel::CreateOperatorByCNode(cast);
  auto ret = cpu_kernel->Init_(op, args.inputs, args.outputs);
  if (!ret) {
    MS_LOG(EXCEPTION) << trace::DumpSourceLines(cast);
  }
  if (cpu_kernel->Resize(args.inputs, args.outputs, kernel::GetKernelDepends(cast)) == kernel::KRET_RESIZE_FAILED) {
    MS_LOG(EXCEPTION) << "CPU kernel op [" << cast->fullname_with_scope() << "] Resize failed.";
  }
  AnfAlgo::SetKernelMod(cpu_kernel, cast.get());
  return cast;
}
}  // namespace opt
}  // namespace mindspore
