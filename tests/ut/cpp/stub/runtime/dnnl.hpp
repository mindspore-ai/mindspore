/*******************************************************************************
 * Copyright 2016-2021 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef TESTS_UT_CPP_STUB_RUNTIME_DNNL_HPP
#define TESTS_UT_CPP_STUB_RUNTIME_DNNL_HPP

/// oneDNN namespace
namespace dnnl {
/// @copydoc dnnl_cpu_isa_t
enum class cpu_isa {
  /// @copydoc dnnl_cpu_isa_all
  all = 0x0,
  /// @copydoc dnnl_cpu_isa_sse41
  sse41 = 0x1,
  /// @copydoc dnnl_cpu_isa_avx
  avx = 0x3,
  /// @copydoc dnnl_cpu_isa_avx2
  avx2 = 0x7,
  /// @copydoc dnnl_cpu_isa_avx512_mic
  avx512_mic = 0xf,
  /// @copydoc dnnl_cpu_isa_avx512_mic_4ops
  avx512_mic_4ops = 0x1f,
  /// @copydoc dnnl_cpu_isa_avx512_core
  avx512_core = 0x27,
  /// @copydoc dnnl_cpu_isa_avx512_core_vnni
  avx512_core_vnni = 0x67,
  /// @copydoc dnnl_cpu_isa_avx512_core_bf16
  avx512_core_bf16 = 0xe7,
  /// @copydoc dnnl_cpu_isa_avx512_core_amx
  avx512_core_amx = 0x3e7,
  /// @copydoc dnnl_cpu_isa_avx2_vnni
  avx2_vnni = 0x407,
};

/// @copydoc dnnl_get_effective_cpu_isa()
inline cpu_isa get_effective_cpu_isa() { return cpu_isa::avx512_core; }
}  // namespace dnnl
#endif