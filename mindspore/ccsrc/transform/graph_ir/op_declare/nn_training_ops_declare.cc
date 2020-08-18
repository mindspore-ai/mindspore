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

#include "transform/graph_ir/op_declare/nn_training_ops_declare.h"

namespace mindspore::transform {
// ApplyMomentum
INPUT_MAP(ApplyMomentum) = {
  {1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(lr)}, {4, INPUT_DESC(grad)}, {5, INPUT_DESC(momentum)}};
ATTR_MAP(ApplyMomentum) = {{"use_nesterov", ATTR_DESC(use_nesterov, AnyTraits<bool>())},
                           {"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyMomentum) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ApplyMomentum, kNameApplyMomentum, ADPT_DESC(ApplyMomentum))

// LarsV2Update
INPUT_MAP(LarsV2Update) = {{1, INPUT_DESC(w)},
                           {2, INPUT_DESC(g)},
                           {3, INPUT_DESC(w_square_sum)},
                           {4, INPUT_DESC(g_square_sum)},
                           {5, INPUT_DESC(weight_decay)},
                           {6, INPUT_DESC(learning_rate)}};
ATTR_MAP(LarsV2Update) = {{"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())},
                          {"hyperpara", ATTR_DESC(hyperpara, AnyTraits<float>())},
                          {"use_clip", ATTR_DESC(use_clip, AnyTraits<bool>())}};
OUTPUT_MAP(LarsV2Update) = {{0, OUTPUT_DESC(g_new)}};
REG_ADPT_DESC(LarsV2Update, kNameLARSUpdate, ADPT_DESC(LarsV2Update))

// ApplyAdam
INPUT_MAP(ApplyAdam) = {{1, INPUT_DESC(var)},         {2, INPUT_DESC(m)},           {3, INPUT_DESC(v)},
                        {4, INPUT_DESC(beta1_power)}, {5, INPUT_DESC(beta2_power)}, {6, INPUT_DESC(lr)},
                        {7, INPUT_DESC(beta1)},       {8, INPUT_DESC(beta2)},       {9, INPUT_DESC(epsilon)},
                        {10, INPUT_DESC(grad)}};
ATTR_MAP(ApplyAdam) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())},
                       {"use_nesterov", ATTR_DESC(use_nesterov, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyAdam) = {{0, OUTPUT_DESC(var)}};

// ApplyAdamD
INPUT_MAP(ApplyAdamD) = {{1, INPUT_DESC(var)},         {2, INPUT_DESC(m)},           {3, INPUT_DESC(v)},
                         {4, INPUT_DESC(beta1_power)}, {5, INPUT_DESC(beta2_power)}, {6, INPUT_DESC(lr)},
                         {7, INPUT_DESC(beta1)},       {8, INPUT_DESC(beta2)},       {9, INPUT_DESC(epsilon)},
                         {10, INPUT_DESC(grad)}};
ATTR_MAP(ApplyAdamD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())},
                        {"use_nesterov", ATTR_DESC(use_nesterov, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyAdamD) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(m)}, {2, OUTPUT_DESC(v)}};
#ifdef ENABLE_GE
REG_ADPT_DESC(ApplyAdamD, kNameApplyAdam, ADPT_DESC(ApplyAdamD))
#else
REG_ADPT_DESC(ApplyAdam, kNameApplyAdam, ADPT_DESC(ApplyAdam))
#endif

// SGD
INPUT_MAP(SGD) = {{1, INPUT_DESC(parameters)}, {2, INPUT_DESC(gradient)}, {3, INPUT_DESC(learning_rate)},
                  {4, INPUT_DESC(accum)},      {5, INPUT_DESC(momentum)}, {6, INPUT_DESC(stat)}};
ATTR_MAP(SGD) = {{"dampening", ATTR_DESC(dampening, AnyTraits<float>())},
                 {"weight_decay", ATTR_DESC(weight_decay, AnyTraits<float>())},
                 {"nesterov", ATTR_DESC(nesterov, AnyTraits<bool>())}};
OUTPUT_MAP(SGD) = {{0, OUTPUT_DESC(parameters)}};
REG_ADPT_DESC(SGD, kNameSGD, ADPT_DESC(SGD))

// SparseApplyAdagradD
INPUT_MAP(SparseApplyAdagradD) = {
  {1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(grad)}, {4, INPUT_DESC(indices)}};
ATTR_MAP(SparseApplyAdagradD) = {{"lr", ATTR_DESC(lr, AnyTraits<float>())},
                                 {"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(SparseApplyAdagradD) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(SparseApplyAdagradD, kNameSparseApplyAdagrad, ADPT_DESC(SparseApplyAdagradD))

// ApplyProximalAdagradD
INPUT_MAP(ApplyProximalAdagradD) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(accum)}, {3, INPUT_DESC(lr)},
                                    {4, INPUT_DESC(l1)},  {5, INPUT_DESC(l2)},    {6, INPUT_DESC(grad)}};
ATTR_MAP(ApplyProximalAdagradD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyProximalAdagradD) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(accum)}};
REG_ADPT_DESC(ApplyProximalAdagradD, kNameApplyProximalAdagrad, ADPT_DESC(ApplyProximalAdagradD))

// SparseApplyFtrlD
INPUT_MAP(SparseApplyFtrlD) = {{1, INPUT_DESC(var)},
                               {2, INPUT_DESC(accum)},
                               {3, INPUT_DESC(linear)},
                               {4, INPUT_DESC(grad)},
                               {5, INPUT_DESC(indices)}};
ATTR_MAP(SparseApplyFtrlD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())},
                              {"lr", ATTR_DESC(lr, AnyTraits<float>())},
                              {"l1", ATTR_DESC(l1, AnyTraits<float>())},
                              {"l2", ATTR_DESC(l2, AnyTraits<float>())},
                              {"lr_power", ATTR_DESC(lr_power, AnyTraits<float>())}};
OUTPUT_MAP(SparseApplyFtrlD) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(SparseApplyFtrlD, kNameSparseApplyFtrlD, ADPT_DESC(SparseApplyFtrlD))

// ApplyFtrlD
INPUT_MAP(ApplyFtrlD) = {{1, INPUT_DESC(var)},  {2, INPUT_DESC(accum)},   {3, INPUT_DESC(linear)},
                         {4, INPUT_DESC(grad)}, {5, INPUT_DESC(lr)},      {6, INPUT_DESC(l1)},
                         {7, INPUT_DESC(l2)},   {8, INPUT_DESC(lr_power)}};
ATTR_MAP(ApplyFtrlD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyFtrlD) = {{0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(accum)}, {2, OUTPUT_DESC(linear)}};
REG_ADPT_DESC(ApplyFtrlD, kNameApplyFtrl, ADPT_DESC(ApplyFtrlD))

// ApplyRMSPropD
INPUT_MAP(ApplyRMSPropD) = {
  {1, INPUT_DESC(var)}, {2, INPUT_DESC(ms)}, {3, INPUT_DESC(mom)}, {4, INPUT_DESC(lr)}, {5, INPUT_DESC(grad)}};
INPUT_ATTR_MAP(ApplyRMSPropD) = {{6, ATTR_DESC(rho, AnyTraits<float>())},
                                 {7, ATTR_DESC(momentum, AnyTraits<float>())},
                                 {8, ATTR_DESC(epsilon, AnyTraits<float>())}};
ATTR_MAP(ApplyRMSPropD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyRMSPropD) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(ApplyRMSPropD, kNameApplyRMSProp, ADPT_DESC(ApplyRMSPropD))

// ApplyCenteredRMSPropD
INPUT_MAP(ApplyCenteredRMSPropD) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(mg)},       {3, INPUT_DESC(ms)},
                                    {4, INPUT_DESC(mom)}, {5, INPUT_DESC(grad)},     {6, INPUT_DESC(lr)},
                                    {7, INPUT_DESC(rho)}, {8, INPUT_DESC(momentum)}, {9, INPUT_DESC(epsilon)}};
ATTR_MAP(ApplyCenteredRMSPropD) = {{"use_locking", ATTR_DESC(use_locking, AnyTraits<bool>())}};
OUTPUT_MAP(ApplyCenteredRMSPropD) = {
  {0, OUTPUT_DESC(var)}, {1, OUTPUT_DESC(mg)}, {2, OUTPUT_DESC(ms)}, {3, OUTPUT_DESC(mom)}};
REG_ADPT_DESC(ApplyCenteredRMSPropD, kNameApplyCenteredRMSProp, ADPT_DESC(ApplyCenteredRMSPropD))
}  // namespace mindspore::transform
