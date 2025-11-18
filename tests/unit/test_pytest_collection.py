"""
Test to prevent pytest collection warnings.

This test ensures that domain classes starting with "Test"
are properly marked with __test__ = False to prevent pytest
from attempting to collect them as test classes.

See: docs/failures/003-pytest-collection-warnings.md
"""
import inspect
import pytest
from pathlib import Path


def get_all_classes_in_module(module):
    """Get all classes defined in a module."""
    return [
        (name, obj) for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
    ]


def test_domain_classes_starting_with_test_are_marked():
    """
    Verify that domain classes starting with 'Test' have __test__ = False.

    This prevents pytest from trying to collect them as test classes,
    which causes collection warnings.
    """
    # Import domain modules
    from app.core.domain.entities import evaluation
    from app.core.domain import exceptions

    modules_to_check = [evaluation, exceptions]
    problematic_classes = []

    for module in modules_to_check:
        classes = get_all_classes_in_module(module)

        for class_name, class_obj in classes:
            # Check if class name starts with "Test"
            if class_name.startswith("Test"):
                # Check if __test__ attribute is set to False
                has_test_attr = hasattr(class_obj, "__test__")
                test_attr_value = getattr(class_obj, "__test__", None)

                if not has_test_attr or test_attr_value is not False:
                    problematic_classes.append({
                        "module": module.__name__,
                        "class": class_name,
                        "has_test_attr": has_test_attr,
                        "test_attr_value": test_attr_value
                    })

    # Assert no problematic classes found
    if problematic_classes:
        error_msg = (
            "Found domain classes starting with 'Test' that lack __test__ = False:\n"
        )
        for cls_info in problematic_classes:
            error_msg += (
                f"  - {cls_info['module']}.{cls_info['class']} "
                f"(has __test__: {cls_info['has_test_attr']}, "
                f"value: {cls_info['test_attr_value']})\n"
            )
        error_msg += (
            "\nFix: Add '__test__ = False' to these classes to prevent "
            "pytest collection warnings.\n"
            "See: docs/failures/003-pytest-collection-warnings.md"
        )
        pytest.fail(error_msg)


def test_pytest_collection_runs_without_warnings():
    """
    Verify pytest collection completes without warnings.

    This is a meta-test that runs pytest collection in a subprocess
    and checks for warning messages.
    """
    import subprocess
    import sys

    # Run pytest --collect-only and capture output
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )

    # Check for collection warnings in output
    output = result.stdout + result.stderr

    # Look for specific warning patterns
    warning_patterns = [
        "PytestCollectionWarning",
        "cannot collect test class",
        "because it has a __init__ constructor"
    ]

    found_warnings = []
    for pattern in warning_patterns:
        if pattern in output:
            found_warnings.append(pattern)

    # Assert no collection warnings
    if found_warnings:
        pytest.fail(
            f"Pytest collection produced warnings: {found_warnings}\n"
            f"Output:\n{output}\n\n"
            f"This usually means domain classes starting with 'Test' "
            f"need __test__ = False attribute."
        )


def test_known_domain_classes_have_test_false():
    """
    Explicit test for known domain classes that caused issues.

    This test specifically checks the classes that historically
    caused pytest collection warnings.
    """
    from app.core.domain.entities.evaluation import TestCase
    from app.core.domain.exceptions import TestCaseLoadError

    # TestCase from evaluation.py
    assert hasattr(TestCase, "__test__"), (
        "TestCase class missing __test__ attribute. "
        "Add '__test__ = False' to prevent pytest collection warnings."
    )
    assert TestCase.__test__ is False, (
        f"TestCase.__test__ should be False, got {TestCase.__test__}"
    )

    # TestCaseLoadError from exceptions.py
    assert hasattr(TestCaseLoadError, "__test__"), (
        "TestCaseLoadError class missing __test__ attribute. "
        "Add '__test__ = False' to prevent pytest collection warnings."
    )
    assert TestCaseLoadError.__test__ is False, (
        f"TestCaseLoadError.__test__ should be False, got {TestCaseLoadError.__test__}"
    )
