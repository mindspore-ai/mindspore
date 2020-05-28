# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""BPE."""
import os
import subprocess

ENCODER = "subword-nmt apply-bpe -c {codes} -i {input} -o {output}"
LEARN_DICT = "subword-nmt get-vocab -i {input} -o {dict_path}"


def bpe_encode(codes_path, src_path, output_path, dict_path):
    """
    Do bpe.

    Args:
        codes_path (str): BPE codes file.
        src_path (str): Source text file path.
        output_path (str): Output path.
        dict_path (str): Dict path.
    """
    if not (os.path.isabs(codes_path)
            and os.path.isabs(src_path)
            and os.path.isabs(output_path)
            and os.path.isabs(dict_path)):
        raise ValueError("Absolute path is required.")

    if not (os.path.exists(os.path.dirname(codes_path))
            and os.path.exists(os.path.dirname(src_path))
            and os.path.exists(os.path.dirname(output_path))
            and os.path.exists(os.path.dirname(dict_path))):
        raise FileNotFoundError("Dir not found.")

    # Encoding.
    print(f" | Applying BPE encoding.")
    subprocess.call(ENCODER.format(codes=codes_path, input=src_path, output=output_path),
                    shell=True)
    print(f" | Fetching vocabulary from single file.")
    # Learn vocab.
    subprocess.call(LEARN_DICT.format(input=output_path, dict_path=dict_path),
                    shell=True)
