import os
import json
import tempfile
import argparse
import shutil
from unittest.mock import patch
from mindspore.custom_compiler.setup import CustomOOC, get_config


def test_parse_args():
    """
    Feature: Command line argument parsing
    Description: Simulates command line arguments and checks if they are parsed correctly by `get_config`.
    Expectation: Configuration object should have correct paths and `install` flag set to True.
    """
    test_args = ['filename', '-o', 'host_path', '-k', 'kernel_path', '-i']
    with patch('sys.argv', test_args):
        res = get_config()
        assert res.op_host_path == "host_path"
        assert res.op_kernel_path == "kernel_path"
        assert res.install


def test_check_args():
    """
    Feature: Argument validation for CustomOOC.
    Description: Checks if `CustomOOC` validates provided arguments correctly.
    Expectation: The test should pass without any exceptions.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_args = argparse.Namespace(op_host_path=temp_dir, op_kernel_path=temp_dir, soc_version="ascend310p",
                                        ascend_cann_package_path=temp_dir, install_path=temp_dir, install=False)
        custom_ooc = CustomOOC(input_args)
        custom_ooc.check_args()


def test_check_args_op_host():
    """
    Feature: Argument validation for CustomOOC.
    Description: Checks for a `ValueError` if the operation host path does not exist.
    Expectation: A `ValueError` should be raised indicating the non-existence of the operation host path.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_args = argparse.Namespace(op_host_path=temp_dir + "/test", op_kernel_path=temp_dir,
                                        soc_version="ascend310p",
                                        ascend_cann_package_path=temp_dir, install_path=temp_dir)
        try:
            custom_ooc = CustomOOC(input_args)
            custom_ooc.check_args()
        except ValueError as e:
            assert f"op host path [{temp_dir}/test] is not exist" in str(e)


def test_check_args_op_kernel():
    """
    Feature: Argument validation for CustomOOC.
    Description: Checks for a `ValueError` if the operation kernel path does not exist.
    Expectation: A `ValueError` should be raised indicating the non-existence of the operation kernel path.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_args = argparse.Namespace(op_host_path=temp_dir, op_kernel_path=temp_dir + "/test",
                                        soc_version="ascend310p",
                                        ascend_cann_package_path=temp_dir, install_path=temp_dir)
        try:
            custom_ooc = CustomOOC(input_args)
            custom_ooc.check_args()
        except ValueError as e:
            assert f"op kernel path [{temp_dir}/test] is not exist" in str(e)


def test_check_args_soc_version():
    """
    Feature: Validation of supported SOC version
    Description: Ensures that an unsupported SOC version raises a `ValueError`.
    Expectation: A `ValueError` should be raised with a message indicating the unsupported SOC version.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_args = argparse.Namespace(op_host_path=temp_dir, op_kernel_path=temp_dir, soc_version="ascend210p",
                                        ascend_cann_package_path=temp_dir, install_path=temp_dir)
        try:
            custom_ooc = CustomOOC(input_args)
            custom_ooc.check_args()
        except ValueError as e:
            assert "Unsupported soc version" in str(e)


def test_check_args_cann_path():
    """
    Feature: Validation of Ascend CANN package path
    Description: Checks for a `ValueError` if the Ascend CANN package path is not a valid path.
    Expectation: A `ValueError` should be raised with a message indicating the invalid CANN package path.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_args = argparse.Namespace(op_host_path=temp_dir, op_kernel_path=temp_dir, soc_version="ascend310p",
                                        ascend_cann_package_path=temp_dir + "/test", install_path=temp_dir)
        try:
            custom_ooc = CustomOOC(input_args)
            custom_ooc.check_args()
        except ValueError as e:
            assert f"ascend cann package path [{temp_dir}/test] is not valid path" in str(e)


def test_check_args_install_path():
    """
    Feature: Validation of installation path
    Description: Ensures that a `ValueError` is raised if the installation path is not valid.
    Expectation: A `ValueError` should be raised with a message indicating the invalid installation path.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_args = argparse.Namespace(op_host_path=temp_dir, op_kernel_path=temp_dir, soc_version="ascend310p",
                                        ascend_cann_package_path=temp_dir,
                                        install_path=temp_dir + "/test", install=False)
        try:
            custom_ooc = CustomOOC(input_args)
            custom_ooc.check_args()
        except ValueError as e:
            assert f"Install path [{temp_dir}/test] is not valid path" in str(e)


def test_check_args_install():
    """
    Feature: Checking the installation path environment variable
    Description: Verifies that a `ValueError` is raised if the `ASCEND_OPP_PATH` environment variable is not set.
    Expectation: A `ValueError` should be raised with a message indicating that the installation path cannot be found.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_args = argparse.Namespace(op_host_path=temp_dir, op_kernel_path=temp_dir, soc_version="ascend310p",
                                        ascend_cann_package_path=temp_dir,
                                        install=True, install_path="")
        os.putenv('ASCEND_OPP_PATH', '')
        try:
            custom_ooc = CustomOOC(input_args)
            custom_ooc.check_args()
        except ValueError as e:
            assert "Can not find install path" in str(e)


def test_install_custom():
    """
    Feature: Installing custom operations
    Description: Checks for a `RuntimeError` if the installation of custom operations fails.
    Expectation: A `RuntimeError` should be raised with a message indicating installation failure.
    """
    input_args = argparse.Namespace(clear=True, install=True, install_path="/tmp")
    try:
        custom_ooc = CustomOOC(input_args)
        custom_ooc.install_custom()
    except RuntimeError as e:
        assert "Install failed" in str(e)


class TestAscendCCompile():
    def setup(self):
        script_path, _ = os.path.split(__file__)
        self.dest_dir = script_path + "/../../../../mindspore/python/mindspore/custom_compiler"
        self.custom_project = os.path.join(self.dest_dir, 'CustomProject')
        cmake_preset = os.path.join(script_path, "CMakePresets.json")
        os.makedirs(self.custom_project, exist_ok=True)
        os.makedirs(os.path.join(self.custom_project, "op_host"), exist_ok=True)
        os.makedirs(os.path.join(self.custom_project, "op_kernel"), exist_ok=True)
        shutil.copy(cmake_preset, self.custom_project)

    def teardown(self):
        shutil.rmtree(self.custom_project)

    def test_compile_config(self):
        """
        Feature: Compiling build configuration
        Description: Checks if the `_compile_config` method correctly sets up the build configuration.
        Expectation: The `CMakePresets.json` file should contain the correct values.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            input_args = argparse.Namespace(ascend_cann_package_path=temp_dir, soc_version="ascend310p",
                                            vendor_name="test_case")
            custom_ooc = CustomOOC(input_args)
            custom_ooc.compile_config()
            with open(os.path.join(self.custom_project, 'CMakePresets.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert data['configurePresets'][0]["cacheVariables"]["ASCEND_CANN_PACKAGE_PATH"]["value"] == temp_dir
            assert data['configurePresets'][0]["cacheVariables"]["ASCEND_COMPUTE_UNIT"]["value"] == "ascend310p"
            assert data['configurePresets'][0]["cacheVariables"]["vendor_name"]["value"] == "test_case"

    def test_compile_config_cann_path_1(self):
        """
        Feature: Validating Ascend CANN package path during configuration compilation
        Description: Ensures that a `RuntimeError` is raised if the CANN package path is not valid.
        Expectation: A `RuntimeError` should be raised with a message indicating the invalid path.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            input_args = argparse.Namespace(ascend_cann_package_path=temp_dir + "/test", soc_version="ascend310p",
                                            vendor_name="test_case")
            try:
                custom_ooc = CustomOOC(input_args)
                custom_ooc.compile_config()
            except RuntimeError as e:
                assert f"The path '{temp_dir}' is not a valid path" in str(e)

    def test_compile_config_cann_path_2(self):
        """
        Feature: Checking the existence of the CANN package path
        Description: Verifies that a `RuntimeError` is raised if the CANN package path is not found.
        Expectation: A `RuntimeError` should be raised indicating that the CANN package path cannot be found.
        """
        input_args = argparse.Namespace(ascend_cann_package_path="", soc_version="ascend310p",
                                        vendor_name="test_case")
        os.environ.pop("ASCEND_AICPU_PATH", None)
        try:
            custom_ooc = CustomOOC(input_args)
            custom_ooc.compile_config()
        except ValueError as e:
            assert "Can not find cann package path" in str(e)

    def test_compile_custom(self):
        """
        Feature: Compiling custom operations
        Description: Verifies that a `RuntimeError` is raised if the compilation of custom operations fails.
        Expectation: A `RuntimeError` should be raised with a message indicating compilation failure.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            host_path = os.path.join(temp_dir, "op_host")
            kernel_path = os.path.join(temp_dir, "op_kernel")
            os.mkdir(host_path)
            os.mkdir(kernel_path)
            with open(os.path.join(host_path, 'host.cpp'), 'w') as file:
                file.write('Hello, world!')
            with open(os.path.join(kernel_path, 'kernel.cpp'), 'w') as file:
                file.write('Hello, world!')
            input_args = argparse.Namespace(op_host_path=host_path, op_kernel_path=kernel_path,
                                            soc_version="ascend310p",
                                            ascend_cann_package_path=temp_dir, install_path=temp_dir)
            try:
                custom_ooc = CustomOOC(input_args)
                custom_ooc.compile_custom()
            except RuntimeError as e:
                assert "Compile failed" in str(e)
