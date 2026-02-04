import os
import stat
import unittest
from unittest.mock import patch
import tempfile
from mindspore.profiler.common.path_manager import PathManager
from mindspore.profiler.common.exceptions.exceptions import ProfilerPathErrorException


class TestPathManager(unittest.TestCase):
    """Test cases for PathManager."""
    def setUp(self):
        """Create temporary test directory and file paths."""
        self.temp_dir = tempfile.mkdtemp(prefix='test_profiler_')
        self.test_dir = os.path.join(self.temp_dir, 'test_dir')
        self.test_file = os.path.join(self.test_dir, 'test.txt')

    def tearDown(self):
        """Clean up temporary test directory."""
        if os.path.exists(self.temp_dir):
            os.system(f"rm -rf {self.temp_dir}")

    def test_check_input_directory_path_should_success_when_path_valid(self):
        """Test check_input_directory_path with valid directory path."""
        os.makedirs(self.test_dir)
        PathManager.check_input_directory_path(self.test_dir)

    def test_check_input_directory_path_should_raise_exception_when_path_too_long(self):
        """Test check_input_directory_path with too long path."""
        long_path = "a" * (PathManager.MAX_PATH_LENGTH + 1)
        with self.assertRaises(ProfilerPathErrorException) as cm:
            PathManager.check_input_directory_path(long_path)
        self.assertIn("exceeds the limit", str(cm.exception))

    def test_check_input_directory_path_should_raise_exception_when_path_is_file(self):
        """Test check_input_directory_path with file path."""
        os.makedirs(self.test_dir)
        with open(self.test_file, 'w') as f:
            f.write("test")
        with self.assertRaises(ProfilerPathErrorException) as cm:
            PathManager.check_input_directory_path(self.test_file)
        self.assertIn("is a file path", str(cm.exception))

    def test_check_input_file_path_should_success_when_path_valid(self):
        """Test check_input_file_path with valid file path."""
        os.makedirs(self.test_dir)
        with open(self.test_file, 'w') as f:
            f.write("test")
        PathManager.check_input_file_path(self.test_file)

    def test_check_input_file_path_should_raise_exception_when_path_is_directory(self):
        """Test check_input_file_path with directory path."""
        os.makedirs(self.test_dir)
        with self.assertRaises(ProfilerPathErrorException) as cm:
            PathManager.check_input_file_path(self.test_dir)
        self.assertIn("is a directory path", str(cm.exception))

    def test_check_input_file_path_should_raise_exception_when_size_exceeds_limit(self):
        """Test check_input_file_path with oversized file."""
        os.makedirs(self.test_dir)
        with patch('os.path.getsize') as mock_size:
            mock_size.return_value = PathManager.MAX_FILE_SIZE + 1
            with self.assertRaises(ProfilerPathErrorException) as cm:
                PathManager.check_input_file_path(self.test_file)
            self.assertIn("file size exceeds the limit", str(cm.exception))

    def test_check_path_owner_consistent_should_success_when_owner_valid(self):
        """Test check_path_owner_consistent with valid owner."""
        os.makedirs(self.test_dir)
        PathManager.check_path_owner_consistent(self.test_dir)

    def test_check_path_owner_consistent_should_raise_exception_when_path_not_exists(self):
        """Test check_path_owner_consistent with non-existent path."""
        with self.assertRaises(ProfilerPathErrorException) as cm:
            PathManager.check_path_owner_consistent("/non/existent/path")
        self.assertIn("does not exist", str(cm.exception))

    def test_check_directory_path_writeable_should_success_when_path_writeable(self):
        """Test check_directory_path_writeable with writable directory."""
        os.makedirs(self.test_dir)
        PathManager.check_directory_path_writeable(self.test_dir)

    def test_check_directory_path_writeable_should_raise_exception_when_not_writeable(self):
        """Test check_directory_path_writeable with non-writable directory."""
        os.makedirs(self.test_dir)
        os.chmod(self.test_dir, 0)  # 设置权限为000

        # 模拟非root用户的权限检查
        with patch('os.access') as mock_access:
            mock_access.return_value = False
            with self.assertRaises(ProfilerPathErrorException) as cm:
                PathManager.check_directory_path_writeable(self.test_dir)
            self.assertIn("writeable permission check failed", str(cm.exception))

        # 恢复权限以便清理
        os.chmod(self.test_dir, stat.S_IRWXU)

    def test_make_dir_safety_should_success_when_path_valid(self):
        """Test make_dir_safety with valid path."""
        PathManager.make_dir_safety(self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertTrue(os.path.isdir(self.test_dir))

    def test_make_dir_safety_should_not_raise_when_directory_exists(self):
        """Test make_dir_safety with existing directory."""
        os.makedirs(self.test_dir)
        PathManager.make_dir_safety(self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))

    def test_remove_path_safety_should_success_when_path_exists(self):
        """Test remove_path_safety with existing directory."""
        os.makedirs(self.test_dir)
        PathManager.remove_path_safety(self.test_dir)
        self.assertFalse(os.path.exists(self.test_dir))

    def test_remove_file_safety_should_success_when_file_exists(self):
        """Test remove_file_safety with existing file."""
        os.makedirs(self.test_dir)
        with open(self.test_file, 'w') as f:
            f.write("test")
        PathManager.remove_file_safety(self.test_file)
        self.assertFalse(os.path.exists(self.test_file))

    def test_remove_file_safety_should_not_raise_when_file_not_exists(self):
        """Test remove_file_safety with non-existent file."""
        with patch('mindspore.log.warning') as mock_warning:
            PathManager.remove_file_safety("/non/existent/file.txt")
            mock_warning.assert_called_once_with("The file does not exist: %s", "/non/existent/file.txt")

    def test_remove_path_safety_should_not_raise_when_path_not_exists(self):
        """Test remove_path_safety with non-existent path."""
        with patch('mindspore.log.warning') as mock_warning:
            PathManager.remove_path_safety("/non/existent/path")
            mock_warning.assert_called_once_with("The path does not exist: %s", "/non/existent/path")

    def test_check_input_file_path_should_raise_exception_when_path_contains_invalid_chars(self):
        """Test check_input_file_path with invalid characters in path."""
        invalid_paths = [
            "/test/path/with/file!.txt",  # 感叹号
            "/test/path/with/file@.txt",  # @符号
            "/test/path/with/file#.txt",  # #号
            "/test/path/with/file$.txt",  # 美元符号
            "/test/path/with/file%.txt",  # 百分号
            "/test/path/with/file^.txt",  # 尖号
            "/test/path/with/file&.txt",  # &符号
            "/test/path/with/file*.txt",  # 星号
            "/test/path/with/file(.txt",  # 左括号
            "/test/path/with/file).txt",  # 右括号
            "/test/path/with/file+.txt",  # 加号
            "/test/path/with/file=.txt",  # 等号
            "/test/path/with/file[.txt",  # 左方括号
            "/test/path/with/file].txt",  # 右方括号
            "/test/path/with/file{.txt",  # 左花括号
            "/test/path/with/file}.txt",  # 右花括号
            "/test/path/with/file|.txt",  # 竖线
            "/test/path/with/file;.txt",  # 分号
            "/test/path/with/file:.txt",  # 冒号
            "/test/path/with/file'.txt",  # 单引号
            "/test/path/with/file\".txt", # 双引号
            "/test/path/with/file<.txt",  # 小于号
            "/test/path/with/file>.txt",  # 大于号
            "/test/path/with/file,.txt",  # 逗号
            "/test/path/with/file\\.txt", # 反斜杠
        ]

        for invalid_path in invalid_paths:
            with self.assertRaises(ProfilerPathErrorException) as cm:
                PathManager.check_input_file_path(invalid_path)
            self.assertIn("contains invalid characters", str(cm.exception))

    def test_check_input_directory_path_should_raise_exception_when_path_is_symlink(self):
        """Test check_input_directory_path with symlink path."""
        os.makedirs(self.test_dir)
        symlink_path = os.path.join(self.temp_dir, 'symlink_dir')
        os.symlink(self.test_dir, symlink_path)
        with self.assertRaises(ProfilerPathErrorException) as cm:
            PathManager.check_input_directory_path(symlink_path)
        self.assertIn("is a soft link", str(cm.exception))

    def test_check_input_file_path_should_raise_exception_when_path_is_symlink(self):
        """Test check_input_file_path with symlink path."""
        os.makedirs(self.test_dir)
        with open(self.test_file, 'w') as f:
            f.write("test")
        symlink_path = os.path.join(self.temp_dir, 'symlink_file')
        os.symlink(self.test_file, symlink_path)
        with self.assertRaises(ProfilerPathErrorException) as cm:
            PathManager.check_input_file_path(symlink_path)
        self.assertIn("is a soft link", str(cm.exception))

    def test_check_input_directory_path_should_raise_exception_when_dirname_too_long(self):
        """Test check_input_directory_path with directory name exceeding length limit."""
        long_name = "a" * (PathManager.MAX_FILE_NAME_LENGTH + 1)
        invalid_path = os.path.join(self.temp_dir, long_name)
        with self.assertRaises(ProfilerPathErrorException) as cm:
            PathManager.check_input_directory_path(invalid_path)
        self.assertIn("exceeds the limit", str(cm.exception))

    def test_check_input_file_path_should_raise_exception_when_filename_too_long(self):
        """Test check_input_file_path with file name exceeding length limit."""
        long_name = "a" * (PathManager.MAX_FILE_NAME_LENGTH + 1) + ".txt"
        invalid_path = os.path.join(self.temp_dir, long_name)
        with self.assertRaises(ProfilerPathErrorException) as cm:
            PathManager.check_input_file_path(invalid_path)
        self.assertIn("exceeds the limit", str(cm.exception))

    def test_check_path_owner_consistent_should_raise_exception_when_owner_not_match(self):
        """Test check_path_owner_consistent when path owner does not match current user."""
        os.makedirs(self.test_dir)

        # Mock os.name to ensure we're not on Windows
        with patch('os.name', 'posix'):
            # Mock os.getuid to return a different user id
            with patch('os.getuid') as mock_getuid:
                mock_getuid.return_value = 1000
                # Mock os.stat to return a different owner id
                with patch('os.stat') as mock_stat:
                    class MockStat:
                        st_uid = 1001
                    mock_stat.return_value = MockStat()

                    with self.assertRaises(ProfilerPathErrorException) as cm:
                        PathManager.check_path_owner_consistent(self.test_dir)
                    self.assertIn("owner[1001] does not match the current user[1000]", str(cm.exception))

    def test_should_raise_exception_when_others_writable(self):
        """Test that an exception is raised when file has others writable permissions."""
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | stat.S_IWOTH
            with self.assertRaises(ProfilerPathErrorException) as context:
                PathManager.check_path_is_other_writable("/test/path")
            self.assertIn("others writable permissions", str(context.exception))
            self.assertIn("chmod -R 755", str(context.exception))

    @patch('os.stat')
    @patch('os.getuid')
    def test_should_return_false_when_not_owned_by_owner_or_root(self, mock_getuid, mock_stat):
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 1000
        self.assertTrue(PathManager.check_path_is_owner_or_root('/test/path'))
        mock_stat.return_value.st_uid = 0
        self.assertTrue(PathManager.check_path_is_owner_or_root('/test/path'))
        mock_stat.return_value.st_uid = 1001
        self.assertFalse(PathManager.check_path_is_owner_or_root('/test/path'))

    @patch('os.access')
    def test_should_pass_correct_arguments_to_os_access(self, mock_access):
        """Test properly passes arguments to os.access"""
        test_path = '/test/path'
        mock_access.return_value = True
        PathManager.check_path_is_executable(test_path)
        mock_access.assert_called_once_with(test_path, os.X_OK)


if __name__ == '__main__':
    unittest.main()
