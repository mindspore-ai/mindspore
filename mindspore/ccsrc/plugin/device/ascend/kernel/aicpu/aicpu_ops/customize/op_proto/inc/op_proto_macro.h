/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_AICPU_AICPU_OPS_CUSTOMIZE_OP_PTOTO_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_AICPU_AICPU_OPS_CUSTOMIZE_OP_PTOTO_H_

#include "graph/operator_reg.h"

#define REG_CUST_OP(x) REG_OP(Cust##x)
#define CUST_OP_END_FACTORY_REG(x) OP_END_FACTORY_REG(Cust##x)
#define CUST_IMPLEMT_VERIFIER(op_name, x) IMPLEMT_VERIFIER(Cust##op_name, x)
#define CUST_VERIFY_FUNC_REG(op_name, x) VERIFY_FUNC_REG(Cust##op_name, x)
#define CUST_IMPLEMT_INFERFUNC(op_name, x) IMPLEMT_INFERFUNC(Cust##op_name, x)
#define CUST_INFER_FUNC_REG(op_name, x) INFER_FUNC_REG(Cust##op_name, x)
#define CUST_COMMON_INFER_FUNC_REG(op_name, x) COMMON_INFER_FUNC_REG(Cust##op_name, x)

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_AICPU_AICPU_OPS_CUSTOMIZE_OP_PTOTO_H_
