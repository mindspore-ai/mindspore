# Copyright 2023 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
The configuration module provides various functions to set and get the supported
configuration parameters.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    from mindspore.mindrecord import set_enc_key, set_enc_mode, set_dec_mode
"""

import os
import shutil
import stat
import time

from mindspore import log as logger
from mindspore._c_expression import _encrypt, _decrypt_data
from .core.shardutils import MIN_FILE_SIZE


__all__ = ['set_enc_key',
           'set_enc_mode',
           'set_dec_mode']


# default encode key
ENC_KEY = None
ENC_MODE = "AES-GCM"
DEC_MODE = None
HASH_MODE = None


# the final mindrecord after encode should be like below
# 1. for create new mindrecord
# mindrecord -> enc_mindrecord+'ENCRYPT'
# 2. for read mindrecord
# enc_mindrecord+'ENCRYPT' -> mindrecord


# mindrecord file encode end flag, we will append 'ENCRYPT' to the end of file
ENCRYPT_END_FLAG = str('ENCRYPT').encode('utf-8')


# directory which stored decrypt mindrecord files
DECRYPT_DIRECTORY = ".decrypt_mindrecord"
DECRYPT_DIRECTORY_LIST = []


# time for warning when encrypt/decrypt takes too long time
ENCRYPT_TIME = 0
DECRYPT_TIME = 0
WARNING_INTERVAL = 30   # 30s


def set_enc_key(enc_key):
    """
    Set the encode key.

    Args:
        enc_key (str): Str-type key used for encryption. The valid length is 16, 24, or 32.
            ``None`` indicates that encryption is not enabled.

    Raises:
        ValueError: The input is not str or length error.

    Examples:
        >>> from mindspore.mindrecord import set_enc_key
        >>>
        >>> set_enc_key("0123456789012345")
    """
    global ENC_KEY

    if enc_key is None:
        ENC_KEY = None
        return

    if not isinstance(enc_key, str):
        raise ValueError("The input enc_key is not str.")

    if len(enc_key) not in [16, 24, 32]:
        raise ValueError("The length of input enc_key is not 16, 24, 32.")

    ENC_KEY = enc_key


def _get_enc_key():
    """Get the encode key. If the enc_key is not set, it will return ``None``."""
    return ENC_KEY


def set_enc_mode(enc_mode="AES-GCM"):
    """
    Set the encode mode.

    Args:
        enc_mode (Union[str, function], optional): This parameter is valid only when enc_key is not set to ``None`` .
            Specifies the encryption mode or customized encryption function, currently supports ``"AES-GCM"`` .
            Default: ``"AES-GCM"`` . If it is customized encryption, users need
            to ensure its correctness, the security of the encryption algorithm and raise exceptions when errors occur.

    Raises:
        ValueError: The input is not valid encode mode or callable function.

    Examples:
        >>> from mindspore.mindrecord import set_enc_mode
        >>>
        >>> set_enc_mode("AES-GCM")
    """
    global ENC_MODE

    if callable(enc_mode):
        ENC_MODE = enc_mode
        return

    if not isinstance(enc_mode, str):
        raise ValueError("The input enc_mode is not str.")

    if enc_mode not in ["AES-GCM"]:
        raise ValueError("The input enc_mode is invalid.")

    ENC_MODE = enc_mode


def _get_enc_mode():
    """Get the encode mode. If the enc_mode is not set, it will return default encode mode ``"AES-GCM"``."""
    return ENC_MODE


def set_dec_mode(dec_mode="AES-GCM"):
    """
    Set the decode mode.

    If the built-in `enc_mode` is used and `dec_mode` is not specified, the encryption algorithm specified by `enc_mode`
    is used for decryption. If you are using customized encryption function, you must specify customized decryption
    function at read time.

    Args:
        dec_mode (Union[str, function], optional): This parameter is valid only when enc_key is not set to ``None`` .
            Specifies the decryption mode or customized decryption function, currently supports ``"AES-GCM"`` .
            Default: ``"AES-GCM"`` . ``None`` indicates that decryption
            mode is not defined. If it is customized decryption, users need to ensure its correctness and raise
            exceptions when errors occur.

    Raises:
        ValueError: The input is not valid decode mode or callable function.

    Examples:
        >>> from mindspore.mindrecord import set_dec_mode
        >>>
        >>> set_dec_mode("AES-GCM")
    """
    global DEC_MODE

    if dec_mode is None:
        DEC_MODE = None
        return

    if callable(dec_mode):
        DEC_MODE = dec_mode
        return

    if not isinstance(dec_mode, str):
        raise ValueError("The input dec_mode is not str.")

    if dec_mode not in ["AES-GCM"]:
        raise ValueError("The input dec_mode is invalid.")

    DEC_MODE = dec_mode


def _get_dec_mode():
    """Get the decode mode. If the dec_mode is not set, it will return encode mode."""
    if DEC_MODE is None:
        if callable(ENC_MODE):
            raise RuntimeError("You use custom encryption, so you must also define custom decryption.")
        return ENC_MODE

    return DEC_MODE


def _get_enc_mode_as_str():
    """Get the encode mode as string. The length of mode should be 7."""
    valid_enc_mode = ""
    if callable(ENC_MODE):
        valid_enc_mode = "UDF-ENC"  # "UDF-ENC"
    else:
        valid_enc_mode = ENC_MODE

    if len(valid_enc_mode) != 7:
        raise RuntimeError("The length of enc_mode string is not 7.")

    return str(valid_enc_mode).encode('utf-8')


def _get_dec_mode_as_str():
    """Get the decode mode as string. The length of mode should be 7."""
    valid_dec_mode = ""

    if DEC_MODE is None:
        if callable(ENC_MODE):
            raise RuntimeError("You use custom encryption, so you must also define custom decryption.")
        valid_dec_mode = ENC_MODE   # "AES-GCM"
    elif callable(DEC_MODE):
        valid_dec_mode = "UDF-ENC"  # "UDF-ENC"
    else:
        valid_dec_mode = DEC_MODE

    if len(valid_dec_mode) != 7:
        raise RuntimeError("The length of enc_mode string is not 7.")

    return str(valid_dec_mode).encode('utf-8')


def encrypt(filename, enc_key, enc_mode):
    """Encrypt the file and the original file will be deleted"""
    if not os.path.exists(filename):
        raise RuntimeError("The input: {} is not exists.".format(filename))

    if not os.path.isfile(filename):
        raise RuntimeError("The input: {} should be a regular file.".format(filename))

    logger.info("Begin to encrypt file: {}.".format(filename))
    start = time.time()

    offset = 64 * 1024 * 1024    ## read the offset 64M
    current_offset = 0           ## use this to seek file
    file_size = os.path.getsize(filename)

    with open(filename, 'rb') as f:
        # create new encrypt file
        encrypt_filename = filename + ".encrypt"
        with open(encrypt_filename, 'wb') as f_encrypt:
            try:
                if callable(enc_mode):
                    enc_mode(f, file_size, f_encrypt, enc_key)
                else:
                    # read the file with offset and do encrypt
                    # original mindrecord file like:
                    # |64M|64M|64M|64M|...
                    # encrypted mindrecord file like:
                    # len+encrypt_data|len+encrypt_data|len+encrypt_data|...|0|enc_mode|ENCRYPT_END_FLAG
                    while True:
                        if file_size - current_offset >= offset:
                            read_size = offset
                        elif file_size - current_offset > 0:
                            read_size = file_size - current_offset
                        else:
                            # have read the entire file
                            break

                        try:
                            f.seek(current_offset)
                        except Exception as exc:  # pylint: disable=W0703
                            f.close()
                            f_encrypt.close()
                            raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}"
                                               .format(filename, current_offset, str(exc))) from exc

                        data = f.read(read_size)
                        encode_data = _encrypt(data, len(data), enc_key, len(enc_key), enc_mode)

                        # write length of data to encrypt file
                        f_encrypt.write(int(len(encode_data)).to_bytes(length=4, byteorder='big', signed=True))

                        # write data to encrypt file
                        f_encrypt.write(encode_data)

                        current_offset += read_size
            except Exception as exc:
                f.close()
                f_encrypt.close()
                os.chmod(encrypt_filename, stat.S_IRUSR | stat.S_IWUSR)
                raise exc

            f.close()

            # writing 0 at the end indicates that all encrypted data has been written.
            f_encrypt.write(int(0).to_bytes(length=4, byteorder='big', signed=True))

            # write enc_mode
            f_encrypt.write(_get_enc_mode_as_str())

            # write ENCRYPT_END_FLAG
            f_encrypt.write(ENCRYPT_END_FLAG)
            f_encrypt.close()

    end = time.time()
    global ENCRYPT_TIME
    ENCRYPT_TIME += end - start
    if ENCRYPT_TIME > WARNING_INTERVAL:
        logger.warning("It takes another " + str(WARNING_INTERVAL) + "s to encrypt the mindrecord file.")
        ENCRYPT_TIME = ENCRYPT_TIME - WARNING_INTERVAL

    # change the file mode
    if os.path.exists(encrypt_filename):
        os.chmod(encrypt_filename, stat.S_IRUSR | stat.S_IWUSR)

        # move the encrypt file to origin file
        shutil.move(encrypt_filename, filename)

    return True


def _get_encrypt_end_flag(filename):
    """get encrypt end flag from the file"""
    if not os.path.exists(filename):
        raise RuntimeError("The input: {} is not exists.".format(filename))

    if not os.path.isfile(filename):
        raise RuntimeError("The input: {} should be a regular file.".format(filename))

    # get the file size first
    file_size = os.path.getsize(filename)
    offset = file_size - len(ENCRYPT_END_FLAG)

    with open(filename, 'rb') as f:
        # get the encrypt end flag which is 'ENCRYPT'
        try:
            f.seek(offset)
        except Exception as exc:  # pylint: disable=W0703
            f.close()
            raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}"
                               .format(filename, offset, str(exc))) from exc

        data = f.read(len(ENCRYPT_END_FLAG))
        f.close()

        return data


def _get_enc_mode_from_file(filename):
    """get encrypt end flag from the file"""
    if not os.path.exists(filename):
        raise RuntimeError("The input: {} is not exists.".format(filename))

    if not os.path.isfile(filename):
        raise RuntimeError("The input: {} should be a regular file.".format(filename))

    # get the file size first
    file_size = os.path.getsize(filename)
    offset = file_size - len(ENCRYPT_END_FLAG) - 7

    with open(filename, 'rb') as f:
        # get the encrypt end flag which is 'ENCRYPT'
        try:
            f.seek(offset)
        except Exception as exc:  # pylint: disable=W0703
            f.close()
            raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}"
                               .format(filename, offset, str(exc))) from exc

        # read the enc_mode str which length is 7
        data = f.read(7)
        f.close()

        return data


def decrypt(filename, enc_key, dec_mode):
    """decrypt the file by enc_key and dec_mode"""
    if not os.path.exists(filename):
        raise RuntimeError("The input: {} is not exists.".format(filename))

    if not os.path.isfile(filename):
        raise RuntimeError("The input: {} should be a regular file.".format(filename))

    whole_file_size = os.path.getsize(filename)
    if whole_file_size < MIN_FILE_SIZE:
        raise RuntimeError("Invalid file, the size of mindrecord file: " + str(whole_file_size) +
                           " is smaller than the lower limit: " + str(MIN_FILE_SIZE) +
                           ".\n Please check file path: " + filename +
                           " and use 'FileWriter' to generate valid mindrecord files.")

    # check ENCRYPT_END_FLAG
    stored_encrypt_end_flag = _get_encrypt_end_flag(filename)
    if _get_enc_key() is not None:
        if stored_encrypt_end_flag != ENCRYPT_END_FLAG:
            raise RuntimeError("The mindrecord file is not encrypted. You can set " +
                               "'mindspore.mindrecord.config.set_enc_key(None)' to disable the decryption.")
    else:
        if stored_encrypt_end_flag == ENCRYPT_END_FLAG:
            raise RuntimeError("The mindrecord file is encrypted. You need to configure " +
                               "'mindspore.mindrecord.config.set_enc_key(...)' and " +
                               "'mindspore.mindrecord.config.set_enc_mode(...)' for decryption.")
        return filename

    # check dec_mode with enc_mode
    enc_mode_from_file = _get_enc_mode_from_file(filename)
    if enc_mode_from_file != _get_dec_mode_as_str():
        raise RuntimeError("Failed to decrypt data, please check if enc_key and enc_mode / dec_mode is valid.")

    logger.info("Begin to decrypt file: {}.".format(filename))
    start = time.time()

    file_size = os.path.getsize(filename) - len(ENCRYPT_END_FLAG)

    global DECRYPT_DIRECTORY_LIST  # pylint: disable=global-variable-not-assigned

    with open(filename, 'rb') as f:
        real_path_filename = os.path.realpath(filename)
        parent_dir = os.path.dirname(real_path_filename)
        only_filename = os.path.basename(real_path_filename)
        current_decrypt_dir = os.path.join(parent_dir, DECRYPT_DIRECTORY)
        if not os.path.exists(current_decrypt_dir):
            os.mkdir(current_decrypt_dir)
            os.chmod(current_decrypt_dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            logger.info("Create directory: {} to store decrypt mindrecord files."
                        .format(os.path.join(parent_dir, DECRYPT_DIRECTORY)))

        if current_decrypt_dir not in DECRYPT_DIRECTORY_LIST:
            DECRYPT_DIRECTORY_LIST.append(current_decrypt_dir)
            logger.warning("The decrypt mindrecord file will be stored in [" + current_decrypt_dir + "] directory. "
                           "If you don't use it anymore after train / eval, you need to delete it manually.")

        # create new decrypt file
        decrypt_filename = os.path.join(current_decrypt_dir, only_filename)
        if os.path.isfile(decrypt_filename):
            # the file which had been decrypted early maybe update by user, so we remove the old decrypted one
            os.remove(decrypt_filename)

        with open(decrypt_filename, 'wb+') as f_decrypt:
            try:
                if callable(dec_mode):
                    dec_mode(f, file_size, f_decrypt, enc_key)
                else:
                    # read the file and do decrypt
                    # encrypted mindrecord file like:
                    # len+encrypt_data|len+encrypt_data|len+encrypt_data|...|0|enc_mode|ENCRYPT_END_FLAG
                    current_offset = 0           ## use this to seek file
                    length = int().from_bytes(f.read(4), byteorder='big', signed=True)
                    while length != 0:
                        # current_offset is the encrypted data
                        current_offset += 4
                        try:
                            f.seek(current_offset)
                        except Exception as exc:  # pylint: disable=W0703
                            f.close()
                            raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}"
                                               .format(filename, current_offset, str(exc))) from exc

                        data = f.read(length)
                        decode_data = _decrypt_data(data, len(data), enc_key, len(enc_key), dec_mode)

                        if decode_data is None:
                            raise RuntimeError("Failed to decrypt data, " +
                                               "please check if enc_key and enc_mode / dec_mode is valid.")

                        # write to decrypt file
                        f_decrypt.write(decode_data)

                        # current_offset is the length of next encrypted data block
                        current_offset += length
                        try:
                            f.seek(current_offset)
                        except Exception as exc:  # pylint: disable=W0703
                            f.close()
                            raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}"
                                               .format(filename, current_offset, str(exc))) from exc

                        length = int().from_bytes(f.read(4), byteorder='big', signed=True)
            except Exception as exc:
                f.close()
                f_decrypt.close()
                os.chmod(decrypt_filename, stat.S_IRUSR | stat.S_IWUSR)
                raise exc

            f.close()
            f_decrypt.close()

    end = time.time()
    global DECRYPT_TIME
    DECRYPT_TIME += end - start
    if DECRYPT_TIME > WARNING_INTERVAL:
        logger.warning("It takes another " + str(WARNING_INTERVAL) + "s to decrypt the mindrecord file.")
        DECRYPT_TIME = DECRYPT_TIME - WARNING_INTERVAL

    # change the file mode
    if os.path.exists(decrypt_filename):
        os.chmod(decrypt_filename, stat.S_IRUSR | stat.S_IWUSR)

    return decrypt_filename
