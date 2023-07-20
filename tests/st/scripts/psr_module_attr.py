#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse

# command line parameters
parser = argparse.ArgumentParser()
parser.add_argument("--module_list", dest="module_list", required=True,
                    help='module list', type=str)
parser.add_argument("--case_root", dest="case_root", required=True,
                    help='case root', type=str)
parser.add_argument("--result_file", dest="result_file", required=True,
                    help='result file', type=str)


class PsrModuleAttr:
    def __init__(self, case_root, result_file, module_list):
        self.case_root = case_root
        self.module_list = json.loads(module_list)
        self.result_file = result_file

        self.surpport_run_case_mod = ["env_card", "env_onecard", "env_single", "env_cluster"]

    @staticmethod
    def get_case_file_pytestmark(case_root, package, case_name, class_name=None, func_name=None):
        """
         get pytestmark info from testcase name, function/class name
        """
        root_path = os.path.abspath(os.path.join(package, ".."))
        package_name = package.split(os.sep)[-1]

        init_py = os.path.join(package, "__init__.py")
        if not os.path.exists(init_py):
            fd = os.open(init_py, os.O_WRONLY|os.O_CREAT, 0o644)
            os.write(fd, str.encode(""))
            os.close(fd)

        sys.path.insert(0, case_root)
        sys.path.insert(0, package)
        sys.path.insert(0, root_path)

        case_name_pkg = __import__(package_name + "." + case_name, fromlist=case_name.split(".")[-1:])

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
                # none attr stype
                if name != "attr":
                    pytestmark_info.append(name)
                else:
                    pytestmark_info.append(list(mark.kwargs.values())[0])

        return pytestmark_info

    def parser_pytest_mark_info(self, pytestmark, package, case_name,
                                test_object_name):
        """
        parse mark as fixed format, only parse "run_case_mod" , "env_type", "component", "feature"
        """
        pytest_mark_info = {}
        is_parser_mark_err = False
        for mark_info in pytestmark:
            if isinstance(mark_info, (str,)):
                mark_infos = [mark_info]
            elif isinstance(mark_info, (list, tuple)):
                mark_infos = mark_info
            else:
                mark_infos = list()

            for tmp_mark_info in mark_infos:
                if tmp_mark_info in self.surpport_run_case_mod:
                    if not pytest_mark_info.get("run_case_mod"):
                        pytest_mark_info["run_case_mod"] = tmp_mark_info
                    else:
                        print('{0}/{1}:{2} env_type can only be one of env_onecard","env_single","env_cluster"' \
                              .format(package, case_name, test_object_name))
                        is_parser_mark_err = True

                elif str(tmp_mark_info).startswith("platform_"):
                    if not pytest_mark_info.get("env_type"):
                        env_type = list()
                        pytest_mark_info["env_type"] = env_type
                    pytest_mark_info.get("env_type").append(tmp_mark_info)

                elif str(tmp_mark_info).startswith("component_"):
                    if not pytest_mark_info.get("component"):
                        pytest_mark_info["component"] = tmp_mark_info
                elif str(tmp_mark_info).startswith("feature_"):
                    if not pytest_mark_info.get("feature"):
                        pytest_mark_info["feature"] = tmp_mark_info

        # when env_type not in ["env_onecard", "env_single", "env_cluster"], report error
        if not pytest_mark_info.get("run_case_mod"):
            print('{0}/{1}:{2} env_type must be one of "env_onecard", "env_single", "env_cluster"' \
                  .format(package, case_name, test_object_name))
            is_parser_mark_err = True

        if is_parser_mark_err:
            return None
        return pytest_mark_info

    def get_function_based_cases(self, package, case_name, function_list):
        attrs = {"package": package, "case_name": case_name, "function_list": [], "class": {}}
        has_psr_func_list = list()
        for function in function_list:
            # func_name has special keyword, in pytest parametrize case, do trunction
            if str(function).__contains__('[') or str(function).__contains__('('):
                print("abnormal func_name {} has been to be filter".format(function))
                func_name = function.split('[')[0].split('(')[0]
                if func_name in has_psr_func_list:
                    continue
                has_psr_func_list.append(func_name)
            else:
                func_name = function

            pytestmark = self.get_case_file_pytestmark(self.case_root, package, case_name, func_name=func_name)
            function_attr = {"function": func_name}
            pytest_mark_info = self.parser_pytest_mark_info(pytestmark, package, case_name, func_name)
            if pytest_mark_info is None:
                attrs_list.append(None)
                continue
            function_attr.update(pytest_mark_info)
            attrs.get("function_list").append(function_attr)
        return attrs

    def get_class_based_cases(self, package, case_name, function_list, class_info):
        attrs = {"package": package, "case_name": case_name, "function_list": [], "class": class_info}
        # class based testcase satisfying condition
        pytestmark = self.get_case_file_pytestmark(self.case_root, package, case_name, class_name=class_info)
        if pytestmark:
            tmp_attrs = {}
            tmp_class_info = {"class_name": class_info}
            pytest_mark_info = self.parser_pytest_mark_info(pytestmark, package, case_name, class_info)
            if pytest_mark_info is None:
                attrs_list.append(None)
                return None
            tmp_attrs.update(pytest_mark_info)
            tmp_class_info.update(pytest_mark_info)
            attrs["class"] = tmp_class_info
            for function in function_list:
                function_attr = {"function": function}
                function_attr.update(tmp_attrs)
                attrs.get("function_list").append(function_attr)
        # function based testcase satisfying condition
        else:
            tmp_class_info = {"class_name": class_info}
            has_psr_func_list = list()
            for function in function_list:
                # func_name has special keyword, in pytest parametrize case, do trunction
                if str(function).__contains__('[') or str(function).__contains__('('):
                    print("abnormal func_name {} has been to be filter".format(function))
                    func_name = function.split('[')[0].split('(')[0]
                    if func_name in has_psr_func_list:
                        continue
                    has_psr_func_list.append(func_name)
                else:
                    func_name = function
                pytestmark = self.get_case_file_pytestmark(
                    self.case_root, package, case_name,
                    func_name=func_name, class_name=class_info)
                function_attr = {"function": func_name}
                pytest_mark_info = self.parser_pytest_mark_info(
                    pytestmark, package, case_name, func_name)
                if pytest_mark_info is None:
                    attrs_list.append(None)
                    continue
                tmp_class_info.update(pytest_mark_info)
                function_attr.update(pytest_mark_info)
                attrs.get("function_list").append(function_attr)
                attrs["class"] = tmp_class_info
        return attrs

    def parser_module_attr(self):
        attrs_list = list()
        for module in self.module_list:
            package = module.get("package", "")
            case_name = module.get("module", "").split(".")[0]
            function_list = module.get("function_list", "")
            class_info = module.get("class", "")
            print("parser_module_attr  start")

            # testcase file contains only function based (no class based) testcases
            if not class_info:
                attrs = self.get_function_based_cases(package, case_name, function_list)
            # testcase file contains class based testcases
            else:
                attrs = self.get_class_based_cases(package, case_name, function_list, class_info)
                if attrs is None:
                    continue

            attrs_list.append(attrs)

        return attrs_list

    def main(self):
        attrs_list = self.parser_module_attr()

        if not attrs_list:
            sys.exit(1)

        fd = os.open(self.result_file, os.O_WRONLY|os.O_CREAT, 0o644)
        os.write(fd, str.encode(json.dumps(attrs_list)))
        os.close(fd)

        print("attrs_list : %s" % attrs_list)


if __name__ == '__main__':
    args = parser.parse_args()
    input_args = (args.case_root, args.result_file, args.module_list)

    psr_ins = PsrModuleAttr(*input_args)
    psr_ins.main()
