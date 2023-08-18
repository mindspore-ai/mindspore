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

#include "./kv_cache_mgr.h"
namespace ge {
IMPLEMT_COMMON_INFERFUNC(KVCacheMgrInferShape) {
  TensorDesc past_des = op.GetInputDescByName("past");
  TensorDesc cur_des = op.GetInputDescByName("cur");
  TensorDesc index_des = op.GetInputDescByName("index");

  Shape past_shape = past_des.GetShape();
  Shape cur_shape = cur_des.GetShape();
  Shape index_shape = index_des.GetShape();

  TensorDesc out_des = op.GetOutputDescByName("past");
  out_des.SetShape(past_des.GetShape());
  out_des.SetDataType(past_des.GetDataType());

  (void)op.UpdateOutputDesc("past", out_des);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(KVCacheMgr, KVCacheMgrVerify) {
  TensorDesc past_des = op.GetInputDescByName("past");
  TensorDesc cur_des = op.GetInputDescByName("cur");
  TensorDesc index_des = op.GetInputDescByName("index");
  // check DataType
  DataType input_type_past = past_des.GetDataType();
  DataType input_type_cur = cur_des.GetDataType();
  DataType input_type_index = index_des.GetDataType();

  if (input_type_past != input_type_cur) {
    return GRAPH_FAILED;
  }
  if (input_type_past != DT_FLOAT && input_type_past != DT_INT32 && input_type_past != DT_UINT32 &&
      input_type_past != DT_FLOAT16 && input_type_past != DT_INT16 && input_type_past != DT_UINT16 &&
      input_type_past != DT_INT8 && input_type_past != DT_UINT8) {
    return GRAPH_FAILED;
  }
  if (input_type_index != DT_INT32) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(KVCacheMgr, KVCacheMgrInferShape);
VERIFY_FUNC_REG(KVCacheMgr, KVCacheMgrVerify);
}  // namespace ge
