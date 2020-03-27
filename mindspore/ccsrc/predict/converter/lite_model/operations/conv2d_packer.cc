/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "predict/converter/lite_model/op_attr_packer.h"

namespace mindspore {
namespace predict {
namespace convert {
bool Conv2dPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op) {
  if (c_node_ptr == nullptr || ms_op == nullptr) {
    return false;
  }
  int kernel_group_value = AnfAlgo::GetNodeAttr<int>(c_node_ptr, "group");
  int kernel_channel_value = AnfAlgo::GetNodeAttr<int>(c_node_ptr, "out_channel");
  std::vector<int> kernel_size_value = AnfAlgo::GetNodeAttr<std::vector<int>>(c_node_ptr, "kernel_size");
  std::string kernel_pad_mode_value = AnfAlgo::GetNodeAttr<std::string>(c_node_ptr, "pad_mode");
  int kernel_pad_value = AnfAlgo::GetNodeAttr<int>(c_node_ptr, "pad");
  int kernel_stride_value = AnfAlgo::GetNodeAttr<int>(c_node_ptr, "stride");
  int kernel_dilation_value = AnfAlgo::GetNodeAttr<int>(c_node_ptr, "dilation");
  std::string kernel_data_format_value = AnfAlgo::GetNodeAttr<std::string>(c_node_ptr, "data_format");
  std::unique_ptr<Conv2DT> attr(new Conv2DT());
  MS_EXCEPTION_IF_NULL(attr);
  attr->format = GetAttrFormat(kernel_data_format_value);
  attr->group = kernel_group_value;
  auto in_shape = AnfAlgo::GetPrevNodeOutputInferShape(c_node_ptr, 1);
  if (in_shape.size() != kNCHWSize) {
    return false;
  }
  attr->channelIn = SizeToInt(in_shape[1]);
  attr->channelOut = kernel_channel_value;
  attr->kernelW = kernel_size_value[0];
  attr->kernelH = kernel_size_value[1];
  attr->strideW = kernel_stride_value;
  attr->strideH = kernel_stride_value;
  attr->padMode = GetAttrPadMode(kernel_pad_mode_value);
  attr->padUp = kernel_pad_value;
  attr->padDown = kernel_pad_value;
  attr->padLeft = kernel_pad_value;
  attr->padRight = kernel_pad_value;
  attr->dilateW = kernel_dilation_value;
  attr->dilateH = kernel_dilation_value;
  attr->hasBias = false;
  ms_op->name = c_node_ptr->fullname_with_scope();
  ms_op->attr.type = OpT_Conv2D;
  ms_op->attr.value = attr.release();
  return true;
}
}  // namespace convert
}  // namespace predict
}  // namespace mindspore
