import sys
import argparse
import runpy
import os

from argparse_interceptor import install, uninstall, REGISTRY


class StopExecution(Exception):
    pass


_original_parse_args = argparse.ArgumentParser.parse_args


def patched_parse_args(self, *args, **kwargs):
    REGISTRY.capture(self)
    raise StopExecution()


def introspect_python_tool(entrypoint: str):
    """Introspects a Python-based SNPE tool and returns its argument registry."""
    REGISTRY.clear()

    # Mock potentially failing imports using meta_path
    from unittest.mock import MagicMock
    from importlib.machinery import ModuleSpec

    class MockImporter:
        def find_spec(self, fullname, path, target=None):
            if fullname.startswith("tvm") or fullname.startswith("qti.tvm"):
                return ModuleSpec(fullname, self)
            return None

        def create_module(self, spec):
            m = MagicMock()
            m.__path__ = []
            return m

        def exec_module(self, module):
            pass

    importer = MockImporter()
    sys.meta_path.insert(0, importer)
    
    install()
    argparse.ArgumentParser.parse_args = patched_parse_args

    old_argv = sys.argv
    sys.argv = [entrypoint, "--help"]

    try:
        # Run the script. runpy.run_path is generally safer than import
        results = runpy.run_path(entrypoint, run_name="__main__")
        # If it defines a main() but doesn't call it, we try calling it
        if "main" in results and callable(results["main"]):
            results["main"]()
    except StopExecution:
        pass
    except SystemExit:
        # Many tools call sys.exit() after --help or if args are missing
        pass
    except Exception as e:
        print(f"Error introspecting {entrypoint}: {e}")
    finally:
        sys.argv = old_argv
        argparse.ArgumentParser.parse_args = _original_parse_args
        uninstall()
        if importer in sys.meta_path:
            sys.meta_path.remove(importer)

    return REGISTRY.dump()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python introspect_snpe_tool.py <tool_path>")
        sys.exit(1)
    tool = sys.argv[1]
    spec = introspect_python_tool(tool)

    import json
    print(json.dumps(spec, indent=2))
