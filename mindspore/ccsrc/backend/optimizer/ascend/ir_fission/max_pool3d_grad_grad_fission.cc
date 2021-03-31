/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/ascend/ir_fission/max_pool3d_grad_grad_fission.h"
#include <vector>
#include <memory>
#include <string>
#include "backend/session/anf_runtime_algorithm.h"
#include "frontend/optimizer/opt.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
constexpr size_t kInputNum = 3;
constexpr size_t kFloat16Len = 2;  // size of float16;
namespace {
tensor::TensorPtr CreateTensor(const AnfNodePtr &node) {
  // 1 get attr ksize
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto ksize = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, "kernel_size");
  auto data_format = AnfAlgo::GetNodeAttr<std::string>(cnode, "format");
  if (data_format != kOpFormat_NCDHW) {
    MS_LOG(ERROR) << "MaxPool3DGradGrad only support NCDHW.";
  }
  MS_LOG(DEBUG) << "ksize of MaxPool3DGradGrad:" << ksize;
  int64_t D = ksize[2];
  int64_t H = ksize[3];
  int64_t W = ksize[4];

  // 1 create tensor
  std::vector<int64_t> assist_shape = {1, 1, D, H, W};  // shape:NCDHW
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat16);
  MS_EXCEPTION_IF_NULL(tensor_type);
  tensor::DeviceInfo device_info{kOpFormat_NDC1HWC0, tensor_type};
  tensor::TensorPtr assist_tensor = std::make_shared<tensor::Tensor>(kFloat16->type_id(), assist_shape);
  assist_tensor->set_device_info(device_info);

  // 2 set value of tensor
  auto data_ptr = assist_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  std::vector<float16> half_data;
  int64_t dims = 1 * 1 * D * H * W;
  int64_t counter = dims;
  for (int64_t i = 0; i < dims; i++) {
    half_data.emplace_back(float16(static_cast<float>(counter)));
    counter--;
  }

  auto elem_num = dims * kFloat16Len;
  auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(assist_tensor->data().nbytes()), half_data.data(), elem_num);
  if (ret_code != 0) {
    MS_LOG(ERROR) << "Failed to copy data into Tensor.";
    return nullptr;
  }
  return assist_tensor;
}

ValueNodePtr CreateValueNode(const AnfNodePtr &node) {
  tensor::TensorPtr assist_tensor = CreateTensor(node);
  MS_EXCEPTION_IF_NULL(assist_tensor);
  auto assist_const = std::make_shared<ValueNode>(assist_tensor);
  MS_EXCEPTION_IF_NULL(assist_const);
  auto assist_abstract = assist_tensor->ToAbstract();
  assist_const->set_abstract(assist_abstract);
  auto assist_kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(assist_kernel_info);
  assist_const->set_kernel_info(assist_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder op_builder;
  op_builder.SetOutputsFormat({kOpFormat_NDC1HWC0});
  op_builder.SetOutputsDeviceType({kNumberTypeFloat16});
  AnfAlgo::SetSelectKernelBuildInfo(op_builder.Build(), assist_const.get());
  return assist_const;
}
}  // namespace

const BaseRef MaxPool3DGradGradFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto max_pool3d_grad_grad_prim = std::make_shared<Primitive>(kMaxPool3DGradGradOpName);
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
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(kMaxPool3DGradGradOpName))};
  auto assist_const = CreateValueNode(cnode);
  new_inputs.insert(new_inputs.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
  new_inputs.push_back(assist_const);
  CNodePtr new_cnode = graph->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  new_cnode->set_scope(cnode->scope());
  AnfAlgo::CopyNodeAttrs(cnode, new_cnode);
  if (kernel_graph != nullptr) {
    kernel_graph->AddValueNodeToGraph(assist_const);
    MS_LOG(INFO) << "Split MaxPool3DGradGrad op success.";
  }
  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
