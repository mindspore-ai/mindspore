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

/* runtime基础数据类型声明 */

/* HCCL基础数据类型声明 */
#include "hccl/hcom.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 集合通信域初始化 */
hcclResult_t hcom_init(const char *rank_table, const char *identify) { return HCCL_SUCCESS; }

/* 解析ranktable for python */
hcclResult_t hcom_rank_info_init(const char *rank_table, const char *identify, u32 device_id) { return HCCL_SUCCESS; }

/* 集合通信域销毁 */
hcclResult_t hcom_destroy(void) { return HCCL_SUCCESS; }

/* 绑定model */
hcclResult_t hcom_bind_model(rtModel_t model, rtStream_t stream) { return HCCL_SUCCESS; }

/* 绑解定model */
hcclResult_t hcom_unbind_model(rtModel_t model) { return HCCL_SUCCESS; }

/* allgather功能实现 */
hcclResult_t hcom_all_gather(const char *tag, void *inputPtr, void *outputPtr, u64 inputCount, hcclDataType_t dataType,
                             const char *group, rtStream_t stream) {
  return HCCL_SUCCESS;
}

/* allreduce功能实现 */
hcclResult_t hcom_all_reduce(const char *tag, void *inputPtr, void *outputPtr, u64 count, hcclDataType_t dataType,
                             hcclRedOp_t op, const char *group, rtStream_t stream) {
  return HCCL_SUCCESS;
}

/* broadcas功能实现 */
hcclResult_t hcom_broadcast(const char *tag, void *ptr, u64 count, hcclDataType_t dataType, u32 root, const char *group,
                            rtStream_t stream) {
  return HCCL_SUCCESS;
}
/* reduce_scatter功能实现 */
hcclResult_t hcom_reduce_scatter(const char *tag, void *inputPtr, void *outputPtr, u64 count, hcclDataType_t dataType,
                                 hcclRedOp_t op, const char *group, rtStream_t stream) {
  return HCCL_SUCCESS;
}

/* 获取group内的rank个数 */
hcclResult_t hcom_get_rank_size(const char *group, u32 *rankSize) { return HCCL_SUCCESS; }

/* python获取上云场景内的rank个数 */
hcclResult_t hcom_python_get_rank_size(u32 *rankSize) { return HCCL_SUCCESS; }

/* 获取本rank的id */
hcclResult_t hcom_get_rank_id(const char *group, u32 *rankId) { return HCCL_SUCCESS; }

/* 获取本rank的id */
hcclResult_t hcom_python_get_rank_id(u32 *rankId) { return HCCL_SUCCESS; }

/* 获取本rank的id */
hcclResult_t hcom_get_world_rank_from_group_rank(const char *group, u32 groupRank, u32 *worldRank) {
  return HCCL_SUCCESS;
}

/* 获取通信域的rank个数 */
hcclResult_t hcom_get_group_rank_from_world_rank(u32 worldRank, const char *group, u32 *groupRank) {
  return HCCL_SUCCESS;
}

/* 创建group */
hcclResult_t hcom_create_group(const char *group, u32 rankNum, u32 *rankIds) { return HCCL_SUCCESS; }

/* 销毁group */
hcclResult_t hcom_destroy_group(const char *group) { return HCCL_SUCCESS; }

/* 发送消息 */
hcclResult_t hcom_send(const char *tag, void *inputPtr, u64 count, hcclDataType_t dataType, u32 destRank, u32 srTag,
                       const char *group, rtStream_t stream) {
  return HCCL_SUCCESS;
}

/* 接收消息 */
hcclResult_t hcom_receive(const char *tag, void *outputPtr, u64 count, hcclDataType_t dataType, u32 srcRank, u32 srTag,
                          const char *group, rtStream_t stream) {
  return HCCL_SUCCESS;
}

/* 获取梯度参数切分方案 */
hcclResult_t hcom_get_split_strategy(const char *group, const struct model_feature *feature, u32 maxSegmentNum,
                                     u32 *segmentNum, u32 *segmentIdx) {
  return HCCL_SUCCESS;
}

/* 连通性检测 */
hcclResult_t hcom_connectivity_detection(s32 *result) { return HCCL_SUCCESS; }

hcclResult_t hcom_set_split_strategy_by_index(const char *group, u32 segmentNum, const u32 *IdxList) {
  return HCCL_SUCCESS;
}
hcclResult_t hcom_set_split_strategy_by_size(const char *group, u32 segmentNum, const float *sizeList) {
  return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif
