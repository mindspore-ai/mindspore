/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef NNACL_ASSEMBLY_GLOBAL_H
#define NNACL_ASSEMBLY_GLOBAL_H

// clang-format off
.macro asm_function fname
#ifdef __APPLE__
.globl _\fname
_\fname:
#else
.global \fname
#ifdef __ELF__
.hidden \fname
.type \fname, %function
#endif
\fname:
#endif
.endm

// clang-format off
.macro asm_default_function fname
#ifdef __APPLE__
.globl _\fname
_\fname:
#else
.global \fname
#ifdef __ELF__
.type \fname, %function
#endif
\fname:
#endif
.endm

// clang-format on

#endif  // NNACL_ASSEMBLY_GLOBAL_H
