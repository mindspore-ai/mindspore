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

    from mindspore.mindrecord import set_enc_key, set_enc_mode, set_dec_mode, set_hash_mode
"""

import hashlib
import os
import shutil
import stat
import time

from mindspore import log as logger
from mindspore._c_expression import _encrypt, _decrypt_data
from .shardutils import MIN_FILE_SIZE


__all__ = ['set_enc_key',
           'set_enc_mode',
           'set_dec_mode',
           'set_hash_mode']


# default encode key and hash mode
ENC_KEY = None
ENC_MODE = "AES-GCM"
DEC_MODE = None
HASH_MODE = None


# the final mindrecord after hash check and encode should be like below
# 1. for create new mindrecord: should do hash first, then encode
# mindrecord ->
#               mindrecord+hash_value+len(4bytes)+'HASH' ->
#                                                                enc_mindrecord+'ENCRYPT'
# 2. for read mindrecord, should decode first, then do hash check
# enc_mindrecord+'ENCRYPT' ->
#                             mindrecord+hash_value+len(4bytes)+'HASH'


# mindrecord file encode end flag, we will append 'ENCRYPT' to the end of file
ENCRYPT_END_FLAG = str('ENCRYPT').encode('utf-8')


# mindrecord file hash check flag, we will append hash value+'HASH' to the end of file
HASH_END_FLAG = str('HASH').encode('utf-8')


# length of hash value (4bytes) + 'HASH'
LEN_HASH_WITH_END_FLAG = 4 + len(HASH_END_FLAG)


# directory which stored decrypt mindrecord files
DECRYPT_DIRECTORY = ".decrypt_mindrecord"
DECRYPT_DIRECTORY_LIST = []


# time for warning when encrypt/decrypt or calculate hash takes too long time
CALCULATE_HASH_TIME = 0
VERIFY_HASH_TIME = 0
ENCRYPT_TIME = 0
DECRYPT_TIME = 0
WARNING_INTERVAL = 30   # 30s


def set_enc_key(enc_key):
    """
    Set the encode key.

    Note:
        When the encryption algorithm is ``"SM4-CBC"`` , only 16 bit length key are supported.

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
    global ENC_KEY

    return ENC_KEY


def set_enc_mode(enc_mode="AES-GCM"):
    """
    Set the encode mode.

    Args:
        enc_mode (Union[str, function], optional): This parameter is valid only when enc_key is not set to ``None`` .
            Specifies the encryption mode or customized encryption function, currently supports ``"AES-GCM"``,
            ``"AES-CBC"`` and ``"SM4-CBC"`` . Default: ``"AES-GCM"`` . If it is customized encryption, users need
            to ensure its correctness and raise exceptions when errors occur.

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

    if enc_mode not in ["AES-GCM", "AES-CBC", "SM4-CBC"]:
        raise ValueError("The input enc_mode is invalid.")

    ENC_MODE = enc_mode


def _get_enc_mode():
    """Get the encode mode. If the enc_mode is not set, it will return default encode mode ``"AES-GCM"``."""
    global ENC_MODE

    return ENC_MODE


def set_dec_mode(dec_mode="AES-GCM"):
    """
    Set the decode mode.

    If the built-in `enc_mode` is used and `dec_mode` is not specified, the encryption algorithm specified by `enc_mode`
    is used for decryption. If you are using customized encryption function, you must specify customized decryption
    function at read time.

    Args:
        dec_mode (Union[str, function], optional): This parameter is valid only when enc_key is not set to ``None`` .
            Specifies the decryption mode or customized decryption function, currently supports ``"AES-GCM"``,
            ``"AES-CBC"`` and ``"SM4-CBC"`` . Default: ``"AES-GCM"`` . ``None`` indicates that decryption
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

    if dec_mode not in ["AES-GCM", "AES-CBC", "SM4-CBC"]:
        raise ValueError("The input dec_mode is invalid.")

    DEC_MODE = dec_mode


def _get_dec_mode():
    """Get the decode mode. If the dec_mode is not set, it will return encode mode."""
    global ENC_MODE
    global DEC_MODE

    if DEC_MODE is None:
        if callable(ENC_MODE):
            raise RuntimeError("You use custom encryption, so you must also define custom decryption.")
        return ENC_MODE

    return DEC_MODE


def _get_enc_mode_as_str():
    """Get the encode mode as string. The length of mode should be 7."""
    global ENC_MODE

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
    global ENC_MODE
    global DEC_MODE

    valid_dec_mode = ""

    if DEC_MODE is None:
        if callable(ENC_MODE):
            raise RuntimeError("You use custom encryption, so you must also define custom decryption.")
        valid_dec_mode = ENC_MODE   # "AES-GCM" / "AES-CBC" / "SM4-CBC"
    elif callable(DEC_MODE):
        valid_dec_mode = "UDF-ENC"  # "UDF-ENC"
    else:
        valid_dec_mode = DEC_MODE

    if len(valid_dec_mode) != 7:
        raise RuntimeError("The length of enc_mode string is not 7.")

    return str(valid_dec_mode).encode('utf-8')


def set_hash_mode(hash_mode):
    """
    Set the hash mode to ensure mindrecord file integrity.

    Args:
        hash_mode (Union[str, function]): The parameter is used to specify the hash mode. Specifies the hash
            mode or customized hash function, currently supports ``None``, ``"sha256"``,
            ``"sha384"``, ``"sha512"``, ``"sha3_256"``, ``"sha3_384"``
            and ``"sha3_512"``. ``None`` indicates that hash check is not enabled.

    Raises:
        ValueError: The input is not valid hash mode or callable function.

    Examples:
        >>> from mindspore.mindrecord import set_hash_mode
        >>>
        >>> set_hash_mode("sha256")
    """
    global HASH_MODE

    if hash_mode is None:
        HASH_MODE = None
        return

    if callable(hash_mode):
        HASH_MODE = hash_mode
        return

    if not isinstance(hash_mode, str):
        raise ValueError("The input hash_mode is not str.")

    if hash_mode not in ["sha256", "sha384", "sha512", "sha3_256", "sha3_384", "sha3_512"]:
        raise ValueError("The input hash_mode is invalid.")

    HASH_MODE = hash_mode


def _get_hash_func():
    """Get the hash func by hash mode"""
    global HASH_MODE

    if HASH_MODE is None:
        raise RuntimeError("The HASH_MODE is None, no matching hash function.")

    if callable(HASH_MODE):
        return HASH_MODE

    if HASH_MODE == "sha256":
        return hashlib.sha256()
    if HASH_MODE == "sha384":
        return hashlib.sha384()
    if HASH_MODE == "sha512":
        return hashlib.sha512()
    if HASH_MODE == "sha3_256":
        return hashlib.sha3_256()
    if HASH_MODE == "sha3_384":
        return hashlib.sha3_384()
    if HASH_MODE == "sha3_512":
        return hashlib.sha3_512()
    raise RuntimeError("The HASH_MODE: {} is invalid.".format(HASH_MODE))


def _get_hash_mode():
    """Get the hash check mode."""
    global HASH_MODE

    return HASH_MODE


def calculate_file_hash(filename, whole=True):
    """Calculate the file's hash"""
    if not os.path.exists(filename):
        raise RuntimeError("The input: {} is not exists.".format(filename))

    if not os.path.isfile(filename):
        raise RuntimeError("The input: {} should be a regular file.".format(filename))

    # get the hash func
    m = _get_hash_func()

    f = open(filename, 'rb')

    # get the file size first
    if whole:
        file_size = os.path.getsize(filename)
    else:
        len_hash_offset = os.path.getsize(filename) - LEN_HASH_WITH_END_FLAG
        try:
            f.seek(len_hash_offset)
        except Exception as e:  # pylint: disable=W0703
            f.close()
            raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}"
                               .format(filename, len_hash_offset, str(e)))

        len_hash = int.from_bytes(f.read(4), byteorder='big')  # length of hash value is 4 bytes
        file_size = os.path.getsize(filename) - LEN_HASH_WITH_END_FLAG - len_hash

    offset = 64 * 1024 * 1024    ## read the offset 64M
    current_offset = 0           ## use this to seek file

    # read the file with offset and do sha256 hash
    hash_value = str("").encode('utf-8')
    while True:
        if (file_size - current_offset) >= offset:
            read_size = offset
        elif file_size - current_offset > 0:
            read_size = file_size - current_offset
        else:
            # have read the entire file
            break

        try:
            f.seek(current_offset)
        except Exception as e:  # pylint: disable=W0703
            f.close()
            raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}"
                               .format(filename, current_offset, str(e)))

        data = f.read(read_size)
        if callable(m):
            hash_value = m(data, hash_value)
            if not isinstance(hash_value, bytes):
                raise RuntimeError("User defined hash function should return hash value which is bytes type.")
            if hash_value is None:
                raise RuntimeError("User defined hash function return empty.")
        else:
            m.update(data)

        current_offset += read_size

    f.close()

    if callable(m):
        return hash_value
    return m.digest()


def append_hash_to_file(filename):
    """append the hash value to the end of file"""
    if not os.path.exists(filename):
        raise RuntimeError("The input: {} is not exists.".format(filename))

    if not os.path.isfile(filename):
        raise RuntimeError("The input: {} should be a regular file.".format(filename))

    logger.info("Begin to calculate the hash of the file: {}.".format(filename))
    start = time.time()

    hash_value = calculate_file_hash(filename)

    # append hash value, length of hash value (4bytes) and HASH_END_FLAG to the file
    f = open(filename, 'ab')
    f.write(hash_value)     # append the hash value
    f.write((len(hash_value)).to_bytes(4, byteorder='big', signed=False))  # append the length of hash value
    f.write(HASH_END_FLAG)  # append the HASH_END_FLAG
    f.close()

    end = time.time()
    global CALCULATE_HASH_TIME
    CALCULATE_HASH_TIME += end - start
    if CALCULATE_HASH_TIME > WARNING_INTERVAL:
        logger.warning("It takes another " + str(WARNING_INTERVAL) +
                       "s to calculate the hash value of the mindrecord file.")
        CALCULATE_HASH_TIME = CALCULATE_HASH_TIME - WARNING_INTERVAL

    # change the file mode
    os.chmod(filename, stat.S_IRUSR | stat.S_IWUSR)

    return True


def get_hash_end_flag(filename):
    """get the hash end flag from the file"""
    if not os.path.exists(filename):
        raise RuntimeError("The input: {} is not exists.".format(filename))

    if not os.path.isfile(filename):
        raise RuntimeError("The input: {} should be a regular file.".format(filename))

    # get the file size first
    file_size = os.path.getsize(filename)
    offset = file_size - len(HASH_END_FLAG)
    f = open(filename, 'rb')

    # get the hash end flag which is HASH_END_FLAG
    try:
        f.seek(offset)
    except Exception as e:  # pylint: disable=W0703
        f.close()
        raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}".format(filename, offset, str(e)))

    data = f.read(len(HASH_END_FLAG))
    f.close()

    return data


def get_hash_value(filename):
    """get the file's hash"""
    if not os.path.exists(filename):
        raise RuntimeError("The input: {} is not exists.".format(filename))

    if not os.path.isfile(filename):
        raise RuntimeError("The input: {} should be a regular file.".format(filename))

    # get the file size first
    file_size = os.path.getsize(filename)

    # the hash_value+len(4bytes)+'HASH' is stored in the end of the file
    offset = file_size - LEN_HASH_WITH_END_FLAG
    f = open(filename, 'rb')

    # seek the position for the length of hash value
    try:
        f.seek(offset)
    except Exception as e:  # pylint: disable=W0703
        f.close()
        raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}".format(filename, offset, str(e)))

    len_hash = int.from_bytes(f.read(4), byteorder='big')  # length of hash value is 4 bytes
    hash_value_offset = file_size - len_hash - LEN_HASH_WITH_END_FLAG

    # seek the position for the hash value
    try:
        f.seek(hash_value_offset)
    except Exception as e:  # pylint: disable=W0703
        f.close()
        raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}"
                           .format(filename, hash_value_offset, str(e)))

    # read the hash value
    data = f.read(len_hash)
    f.close()

    return data


def verify_file_hash(filename):
    """Calculate the file hash and compare it with the hash value which is stored in the file"""
    if not os.path.exists(filename):
        raise RuntimeError("The input: {} is not exists.".format(filename))

    if not os.path.isfile(filename):
        raise RuntimeError("The input: {} should be a regular file.".format(filename))

    # verify the hash end flag
    stored_hash_end_flag = get_hash_end_flag(filename)
    if _get_hash_mode() is not None:
        if stored_hash_end_flag != HASH_END_FLAG:
            raise RuntimeError("The mindrecord file is not hashed. You can set " +
                               "'mindspore.mindrecord.config.set_hash_mode(None)' to disable the hash check.")
    else:
        if stored_hash_end_flag == HASH_END_FLAG:
            raise RuntimeError("The mindrecord file is hashed. You need to configure " +
                               "'mindspore.mindrecord.config.set_hash_mode(...)' to enable the hash check.")
        return True

    # get the pre hash value from the end of the file
    stored_hash_value = get_hash_value(filename)

    logger.info("Begin to verify the hash of the file: {}.".format(filename))
    start = time.time()

    # calculate hash by the file
    current_hash = calculate_file_hash(filename, False)

    if stored_hash_value != current_hash:
        raise RuntimeError("The input file: " + filename + " hash check fail. The file may be damaged. "
                           "Or configure a correct hash mode.")

    end = time.time()
    global VERIFY_HASH_TIME
    VERIFY_HASH_TIME += end - start
    if VERIFY_HASH_TIME > WARNING_INTERVAL:
        logger.warning("It takes another " + str(WARNING_INTERVAL) +
                       "s to verify the hash value of the mindrecord file.")
        VERIFY_HASH_TIME = VERIFY_HASH_TIME - WARNING_INTERVAL

    return True


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

    f = open(filename, 'rb')

    # create new encrypt file
    encrypt_filename = filename + ".encrypt"
    f_encrypt = open(encrypt_filename, 'wb')

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
                except Exception as e:  # pylint: disable=W0703
                    f.close()
                    f_encrypt.close()
                    raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}"
                                       .format(filename, current_offset, str(e)))

                data = f.read(read_size)
                encode_data = _encrypt(data, len(data), enc_key, len(enc_key), enc_mode)

                # write length of data to encrypt file
                f_encrypt.write(int(len(encode_data)).to_bytes(length=4, byteorder='big', signed=True))

                # write data to encrypt file
                f_encrypt.write(encode_data)

                current_offset += read_size
    except Exception as e:
        f.close()
        f_encrypt.close()
        os.chmod(encrypt_filename, stat.S_IRUSR | stat.S_IWUSR)
        raise e

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

    f = open(filename, 'rb')

    # get the encrypt end flag which is 'ENCRYPT'
    try:
        f.seek(offset)
    except Exception as e:  # pylint: disable=W0703
        f.close()
        raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}".format(filename, offset, str(e)))

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

    f = open(filename, 'rb')

    # get the encrypt end flag which is 'ENCRYPT'
    try:
        f.seek(offset)
    except Exception as e:  # pylint: disable=W0703
        f.close()
        raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}".format(filename, offset, str(e)))

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

    global DECRYPT_DIRECTORY_LIST

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

    f = open(filename, 'rb')

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

    f_decrypt = open(decrypt_filename, 'wb+')

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
                except Exception as e:  # pylint: disable=W0703
                    f.close()
                    raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}"
                                       .format(filename, current_offset, str(e)))

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
                except Exception as e:  # pylint: disable=W0703
                    f.close()
                    raise RuntimeError("Seek the file: {} to position: {} failed. Error: {}"
                                       .format(filename, current_offset, str(e)))

                length = int().from_bytes(f.read(4), byteorder='big', signed=True)
    except Exception as e:
        f.close()
        f_decrypt.close()
        os.chmod(decrypt_filename, stat.S_IRUSR | stat.S_IWUSR)
        raise e

    f.close()
    f_decrypt.close()

    end = time.time()
    global DECRYPT_TIME
    DECRYPT_TIME += end - start
    if DECRYPT_TIME > WARNING_INTERVAL:
        logger.warning("It takes another " + str(WARNING_INTERVAL) + "s to decrypt the mindrecord file.")
        DECRYPT_TIME = DECRYPT_TIME - WARNING_INTERVAL

    # change the file mode
    os.chmod(decrypt_filename, stat.S_IRUSR | stat.S_IWUSR)

    return decrypt_filename
