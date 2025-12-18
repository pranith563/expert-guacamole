import os
import sys
import json
import argparse
from pathlib import Path

# Import our modular introspectors
from introspect_snpe_tool import introspect_python_tool
from introspect_binary_tool import introspect_binary_tool

def is_python(file_path):
    """Detects if a file is a Python script."""
    if file_path.suffix == ".py":
        return True
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(4096)
            # Check for shebang or common python keywords in the first chunk
            if b"python" in chunk.lower() or b"import " in chunk or b"from " in chunk:
                # Basic check for text vs binary
                if b'\x00' not in chunk:
                    return True
    except Exception:
        pass
    return False

def is_binary(file_path):
    """Detects if a file is an executable binary (ELF or PE)."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header.startswith(b"\x7fELF"): # Linux ELF
                return True
            if header.startswith(b"MZ"):    # Windows PE
                return True
    except Exception:
        pass
    return False

def get_lib_path(bin_dir, sdk_root):
    """Derives the corresponding library path for a given bin directory."""
    # Pattern: <sdk_root>/bin/<arch> -> <sdk_root>/lib/<arch>
    rel_path = os.path.relpath(bin_dir, os.path.join(sdk_root, "bin"))
    lib_path = os.path.join(sdk_root, "lib", rel_path)
    if os.path.exists(lib_path):
        return lib_path
    return None

def main():
    parser = argparse.ArgumentParser(description="Unified SDK Tool Orchestrator")
    parser.add_argument("search_dir", help="Directory to search for tools (e.g., bin/x86_64-windows-msvc)")
    parser.add_argument("--output_dir", default="registries", help="Directory to save generated registries")
    parser.add_argument("--sdk_root", default=".", help="Root directory of the SDK")
    
    args = parser.parse_args()
    
    search_path = Path(args.search_dir)
    output_path = Path(args.output_dir)
    sdk_root = Path(args.sdk_root).absolute()
    
    if not output_path.exists():
        output_path.mkdir(parents=True)
        
    lib_path = get_lib_path(search_path.absolute(), sdk_root)
    print(f"Searching for tools in: {search_path}")
    print(f"Detected library path: {lib_path}")
    
    # Setup PYTHONPATH for Python tools
    python_lib = sdk_root / "lib" / "python"
    if python_lib.exists():
        sys.path.insert(0, str(python_lib))
        os.environ["PYTHONPATH"] = f"{python_lib}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

    tools = [f for f in search_path.iterdir() if f.is_file()]
    
    for tool in tools:
        tool_name = tool.name
        if tool_name.endswith((".bat", ".sh", ".txt", ".md", ".json", ".yaml", ".lock", ".ps1")):
            continue
            
        print(f"\nProcessing tool: {tool_name}")
        registry = None
        
        if is_python(tool):
            print(f"Detected as Python tool.")
            registry = introspect_python_tool(str(tool))
        elif is_binary(tool):
            print(f"Detected as Binary tool.")
            # We can only introspect if the binary matches the current OS
            is_linux_binary = False
            try:
                with open(tool, 'rb') as f:
                    is_linux_binary = f.read(4).startswith(b"\x7fELF")
            except: pass
            
            if (sys.platform == "win32" and is_linux_binary) or (sys.platform != "win32" and not is_linux_binary):
                print(f"Skipping binary {tool_name}: Architecture/OS mismatch.")
                continue
                
            registry = introspect_binary_tool(str(tool), str(lib_path))
        else:
            print(f"Unknown file type for {tool_name}, skipping.")
            continue
            
        if registry:
            out_file = output_path / f"{tool_name}_registry.json"
            with open(out_file, 'w') as f:
                json.dump(registry, f, indent=2)
            print(f"Saved registry to {out_file}")
        else:
            print(f"Failed to capture registry for {tool_name}")

if __name__ == "__main__":
    main()
