/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_LOWLATENCY_COLLECTIVE_COMM_LIB_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_LOWLATENCY_COLLECTIVE_COMM_LIB_H_

#include <map>
#include <memory>
#include <vector>
#include <string>
#include "utils/dlopen_macro.h"
#include "include/runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "plugin/ascend/res_manager/collective/lowlatency_communication_group.h"

#ifndef EXPORT_WRAPPER
#define EXPORT_WRAPPER __attribute__((visibility("default")))
#endif

namespace mindspore {
namespace device {
namespace ascend {
using GroupOptions = mindspore::device::GroupOptions;
constexpr char kLCCLGlobalGroupName[] = "hccl_world_group";

// Low-latency collective communication libaray is implemented on Ascend platform. So some HCCL data types could be
// reused.
class EXPORT_WRAPPER LowlatencyCollectiveCommLib : public CollectiveCommunicationLib {
 public:
  static LowlatencyCollectiveCommLib &GetInstance();

  bool Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) override;

  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks,
                                uint32_t local_group_rank, uint32_t local_group_size,
                                const GroupOptions &config = {}) override;

  int AllReduce(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count, HcclDataType dataType,
                const HcclReduceOp op, const aclrtStream stream);

  int AllGather(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count, HcclDataType dataType,
                const aclrtStream stream);

  int ReduceScatter(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count, HcclDataType dataType,
                    const HcclReduceOp op, const aclrtStream stream);

  int All2All(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count, HcclDataType dataType,
              const aclrtStream stream);

  int Broadcast(const LcclPtr &lccl_ptr, void *buff, size_t count, HcclDataType dataType, int root,
                const aclrtStream stream);

  int MatmulAllReduce(const LcocPtr &lcoc_ptr, const CoCInputPkg &input_pkg, const CoCOutputPkg &output_pkg,
                      void *workspace, const aclrtStream stream);

  int AllGatherMatmul(const LcocPtr &lcoc_ptr, const CoCInputPkg &input_pkg, const CoCOutputPkg &output_pkg,
                      void *workspace, const aclrtStream stream);

  int MatmulReduceScatter(const LcocPtr &lcoc_ptr, const CoCInputPkg &input_pkg, const CoCOutputPkg &output_pkg,
                          void *workspace, const aclrtStream stream);

  // Return lccl communicator so that caller could pass this communicator to communication APIs.
  LcclPtr LcclCommunicator(const std::string &group_name);

  // For lcoc operations, lcoc object should be created for each operator so performance could be optimal.
  LcocPtr CreateLcocForOp(const std::string &group_name);

  // Must set coc parameters before calling lcoc operators.
  void SetParamForLcoc(const LcocPtr &lcoc_ptr, LcalType lcal_type, const CoCTiling &tiling,
                       const CoCParamDesc &param_desc);

  // Lcoc operators need workspace with size returned by lcoc object.
  int64_t GetLcocWorkspaceSize(const LcocPtr &lcoc_ptr);

 private:
  LowlatencyCollectiveCommLib();
  ~LowlatencyCollectiveCommLib() override = default;
};
}  // namespace ascend

extern "C" EXPORT_WRAPPER CollectiveCommunicationLib *lowlatency_communication_lib_instance();
extern "C" EXPORT_WRAPPER int AllReduce(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count,
                                        HcclDataType data_type, const HcclReduceOp reduce_op, const aclrtStream stream);
extern "C" EXPORT_WRAPPER int AllGather(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count,
                                        HcclDataType data_type, const aclrtStream stream);
extern "C" EXPORT_WRAPPER int ReduceScatter(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count,
                                            HcclDataType data_type, const HcclReduceOp reduce_op,
                                            const aclrtStream stream);
extern "C" EXPORT_WRAPPER int All2All(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count,
                                      HcclDataType data_type, const aclrtStream stream);
extern "C" EXPORT_WRAPPER int Broadcast(const LcclPtr &lccl_ptr, void *buff, size_t count, HcclDataType data_type,
                                        int root, const aclrtStream stream);
extern "C" EXPORT_WRAPPER int MatmulAllReduce(const LcocPtr &lcoc_ptr, const CoCInputPkg &input_pkg,
                                              const CoCOutputPkg &output_pkg, void *workspace,
                                              const aclrtStream stream);
extern "C" EXPORT_WRAPPER int AllGatherMatmul(const LcocPtr &lcoc_ptr, const CoCInputPkg &input_pkg,
                                              const CoCOutputPkg &output_pkg, void *workspace,
                                              const aclrtStream stream);
extern "C" EXPORT_WRAPPER int MatmulReduceScatter(const LcocPtr &lcoc_ptr, const CoCInputPkg &input_pkg,
                                                  const CoCOutputPkg &output_pkg, void *workspace,
                                                  const aclrtStream stream);
extern "C" EXPORT_WRAPPER LcclPtr LcclCommunicator(const std::string &group_name);
extern "C" EXPORT_WRAPPER LcocPtr CreateLcocForOp(const std::string &group_name);
extern "C" EXPORT_WRAPPER void SetParamForLcoc(const LcocPtr &lcoc_ptr, LcalType lcal_type, const CoCTiling &tiling,
                                               const CoCParamDesc &param_desc);
extern "C" EXPORT_WRAPPER int64_t GetLcocWorkspaceSize(const LcocPtr &lcoc_ptr);
}  // namespace device
}  // namespace mindspore

ORIGIN_METHOD(AllReduce, int, const LcclPtr &, void *, void *, size_t, HcclDataType, const HcclReduceOp,
              const aclrtStream)
ORIGIN_METHOD(AllGather, int, const LcclPtr &, void *, void *, size_t, HcclDataType, const aclrtStream)
ORIGIN_METHOD(ReduceScatter, int, const LcclPtr &, void *, void *, size_t, HcclDataType, const HcclReduceOp,
              const aclrtStream)
ORIGIN_METHOD(All2All, int, const LcclPtr &, void *, void *, size_t, HcclDataType, const aclrtStream)
ORIGIN_METHOD(Broadcast, int, const LcclPtr &, void *, size_t, HcclDataType, int, const aclrtStream)
ORIGIN_METHOD(MatmulAllReduce, int, const LcocPtr &, const CoCInputPkg &, const CoCOutputPkg &, void *,
              const aclrtStream)
ORIGIN_METHOD(AllGatherMatmul, int, const LcocPtr &, const CoCInputPkg &, const CoCOutputPkg &, void *,
              const aclrtStream)
ORIGIN_METHOD(MatmulReduceScatter, int, const LcocPtr &, const CoCInputPkg &, const CoCOutputPkg &, void *,
              const aclrtStream)
ORIGIN_METHOD(LcclCommunicator, LcclPtr, const std::string &);
ORIGIN_METHOD(CreateLcocForOp, LcocPtr, const std::string &);
ORIGIN_METHOD(SetParamForLcoc, void, const LcocPtr &, LcalType, const CoCTiling &, const CoCParamDesc &);
ORIGIN_METHOD(GetLcocWorkspaceSize, int64_t, const LcocPtr &);
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_LOWLATENCY_COLLECTIVE_COMM_LIB_H_
