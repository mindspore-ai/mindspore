#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse

# command line parameters
parser = argparse.ArgumentParser()
parser.add_argument("--case_root", dest="case_root", required=True,
                    help='test case root', type=str)
parser.add_argument("--package", dest="package",
                    required=True,
                    help='case package', type=str)
parser.add_argument("--case_name", dest="case_name", required=True,
                    help='case name', type=str)
parser.add_argument("--result_file", dest="result_file", required=True,
                    help='full path result file', type=str)
parser.add_argument("--class_name", dest="class_name", required=False,
                    help='class name', type=str, default=None)
parser.add_argument("--func_name", dest="func_name", required=False,
                    help='func name', type=str, default=None)


def get_case_file_pytestmark(case_root, package, case_name, result_file,
                             class_name=None, func_name=None):
    """
    get pytestmark info from testcase name, function/class name
    """
    root_path = os.path.abspath(os.path.join(package, ".."))
    package_name = package.split(os.sep)[-1]

    init_py = os.path.join(package, "__init__.py")
    if not os.path.exists(init_py):
        fd = os.open(init_py, os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o644)
        os.write(fd, str.encode(""))
        os.close(fd)

    sys.path.insert(0, case_root)
    sys.path.insert(0, package)
    sys.path.insert(0, root_path)

    case_name_pkg = __import__(package_name + "." + case_name,
                               fromlist=case_name.split(".")[-1:])

    pytestmark = None
    if func_name and not class_name:
        function_pkg = getattr(case_name_pkg, func_name)
        pytestmark = getattr(function_pkg, "pytestmark")
    elif func_name and class_name:
        class_pkg = getattr(case_name_pkg, class_name)
        function_pkg = getattr(class_pkg, func_name)
        pytestmark = getattr(function_pkg, "pytestmark")
    elif not func_name and class_name:
        class_pkg = getattr(case_name_pkg, class_name)
        pytestmark = getattr(class_pkg, "pytestmark", None)
    else:
        return None

    pytestmark_info = list()
    if pytestmark:
        for mark in pytestmark:
            name = mark.name
            # non attr stype
            if name != "attr":
                pytestmark_info.append(name)
            else:
                pytestmark_info.append(list(mark.kwargs.values())[0])

    if pytestmark_info:
        fd = os.open(result_file, os.O_WRONLY|os.O_CREAT, 0o644)
        os.write(fd, str.encode(json.dumps(pytestmark_info)))
        os.close(fd)

    return pytestmark_info


if __name__ == '__main__':
    args = parser.parse_args()
    input_args = (
        args.case_root, args.package, args.case_name, args.result_file,
        args.class_name, args.func_name,)

    pytest_mark = get_case_file_pytestmark(*input_args)
    print(json.dumps(pytest_mark))
