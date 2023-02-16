/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except i n compliance with the License.
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
#include "plugin/device/ascend/optimizer/ir_fission/max_pool3d_grad_grad_fission.h"
#include <vector>
#include <memory>
#include <string>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/optimizer/opt.h"
#include "backend/common/optimizer/helper.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
constexpr size_t kInputNum = 3;
constexpr size_t kFloat16Len = 2;  // size of float16;
constexpr size_t kKernelSizeNum = 5;
constexpr int64_t kFloat16C0 = 16;
namespace {
tensor::TensorPtr CreateTensor(const AnfNodePtr &node) {
  // 1 get attr ksize
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto ksize = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, "kernel_size");
  auto data_format = common::AnfAlgo::GetNodeAttr<std::string>(cnode, "format");
  if (data_format != kOpFormat_NCDHW) {
    MS_LOG(ERROR) << "MaxPool3DGradGrad only support NCDHW format, but got " << data_format;
  }
  if (ksize.size() != kKernelSizeNum) {
    MS_LOG(EXCEPTION) << "kernel_size of MaxPool3DGradGrad must be five, but got " << ksize
                      << trace::DumpSourceLines(node);
  }
  int64_t d = ksize[kDim2];
  int64_t h = ksize[kDim3];
  int64_t w = ksize[kDim4];

  // 1 create tensor
  std::vector<int64_t> assist_shape = {1, d, 1, h, w, kFloat16C0};  // shape:NDC1HWC0
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat16);
  MS_EXCEPTION_IF_NULL(tensor_type);
  tensor::DeviceInfo device_info{kOpFormat_NDC1HWC0, tensor_type, kOpFormat_NDC1HWC0};
  tensor::TensorPtr assist_tensor = std::make_shared<tensor::Tensor>(kFloat16->type_id(), assist_shape);
  assist_tensor->set_device_info(device_info);

  // 2 set value of tensor
  auto data_ptr = assist_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  const int64_t dims = 1 * 1 * d * h * w * kFloat16C0;
  std::vector<uint16_t> half_data(dims);
  const int64_t maximum = d * h * w;
  int64_t counter = 0;
  for (int64_t i = 0; i < dims; i += kFloat16C0) {
    int64_t base = i - i % kFloat16C0;
    for (int64_t j = 0; j < kFloat16C0; j++) {
      half_data[base + j] = static_cast<uint16_t>(maximum - counter);
    }
    counter++;
  }

  auto elem_num = LongToSize(dims) * kFloat16Len;
  auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(assist_tensor->data().nbytes()),
                           static_cast<void *>(half_data.data()), elem_num);
  if (ret_code != 0) {
    MS_LOG(ERROR)
      << "Failed to copy data into Tensor while creating assist input for MaxPool3dGradGrad op, memcpy_s errorno: "
      << ret_code;
    return nullptr;
  }
  return assist_tensor;
}

ValueNodePtr CreateValueNode(const AnfNodePtr &node) {
  tensor::TensorPtr assist_input_tensor = CreateTensor(node);
  MS_EXCEPTION_IF_NULL(assist_input_tensor);
  auto assist_const = std::make_shared<ValueNode>(assist_input_tensor);
  MS_EXCEPTION_IF_NULL(assist_const);
  auto assist_abstract = assist_input_tensor->ToAbstract();
  assist_const->set_abstract(assist_abstract);
  auto assist_kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(assist_kernel_info);
  assist_const->set_kernel_info(assist_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder op_builder;
  op_builder.SetOutputsFormat({kOpFormat_NDC1HWC0});
  op_builder.SetOutputsDeviceType({kNumberTypeFloat16});
  op_builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(op_builder.Build(), assist_const.get());
  return assist_const;
}
}  // namespace

const BaseRef MaxPool3DGradGradFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto max_pool3d_grad_grad_prim = std::make_shared<Primitive>(kMaxPool3DGradGradDOpName);
  return VectorRef({max_pool3d_grad_grad_prim, Xs});
}

const AnfNodePtr MaxPool3DGradGradFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() != kInputNum + 1) {
    MS_LOG(INFO) << "The node " << cnode->DebugString() << " is not equal to " << kInputNum << " inputs";
    return nullptr;
  }
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(kMaxPool3DGradGradDOpName))};
  auto assist_const = CreateValueNode(cnode);
  (void)new_inputs.insert(new_inputs.cend(), cnode->inputs().cbegin() + 1, cnode->inputs().cend());
  (void)new_inputs.emplace_back(assist_const);
  CNodePtr new_cnode = NewCNode(new_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  new_cnode->set_scope(cnode->scope());
  common::AnfAlgo::CopyNodeAttrs(cnode, new_cnode);
  if (kernel_graph != nullptr) {
    kernel_graph->AddValueNodeToGraph(assist_const);
    MS_LOG(INFO) << "Split MaxPool3DGradGrad op success.";
  }
  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
