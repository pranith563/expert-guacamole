import re
import json
import subprocess
import os
import sys

def parse_snpe_help(text, tool_name="unknown"):
    arguments = []
    
    # flag_pattern matches lines like [ --flag ] or --flag or   --flag
    flag_pattern = re.compile(r'^\s*(?:\[\s*)?(--[a-zA-Z0-9_\-]+(?:,\s*--[a-zA-Z0-9_\-]+)?(?:=<val>)?)(?:\s*\])?\s*(.*)$')
    
    current_arg = None
    
    for line in text.split('\n'):
        match = flag_pattern.match(line)
        if match:
            if current_arg:
                arguments.append(current_arg)
            
            flags_str, desc = match.groups()
            flags = [f.strip() for f in flags_str.split('=')[0].split(',')]
            has_val = '=<val>' in flags_str
            
            current_arg = {
                "flags": flags,
                "dest": flags[0].lstrip('-').replace('-', '_'),
                "required": '[' not in line, # heuristic
                "default": None,
                "action": "StoreAction" if has_val else "StoreTrueAction",
                "nargs": None,
                "choices": None,
                "metavar": "VAL" if has_val else None,
                "help": desc.strip(),
                "parser": tool_name,
                "group": "General Options"
            }
        elif current_arg and line.strip() and not line.startswith('   '):
            current_arg["help"] += " " + line.strip()
        elif not line.strip() and current_arg:
            arguments.append(current_arg)
            current_arg = None
            
    if current_arg:
        arguments.append(current_arg)
    
    return arguments

def introspect_binary_tool(binary_path, lib_path):
    """Introspects a C++ binary SNPE tool and returns its argument registry."""
    env = os.environ.copy()
    tool_name = os.path.basename(binary_path)
    
    if sys.platform == "win32":
        env["PATH"] = lib_path + ";" + env.get("PATH", "")
    else:
        # On Linux, we use LD_LIBRARY_PATH
        env["LD_LIBRARY_PATH"] = lib_path + ":" + env.get("LD_LIBRARY_PATH", "")

    try:
        # Try --help
        result = subprocess.run([binary_path, "--help"], env=env, capture_output=True, text=True)
        help_text = result.stdout if result.stdout.strip() else result.stderr
        
        if not help_text.strip():
             # Try -h
             result = subprocess.run([binary_path, "-h"], env=env, capture_output=True, text=True)
             help_text = result.stdout if result.stdout.strip() else result.stderr
             
        return parse_snpe_help(help_text, tool_name=tool_name)
    except Exception as e:
        print(f"Error running binary {binary_path}: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python introspect_binary_tool.py <binary_path> <lib_path>")
        sys.exit(1)
        
    binary = sys.argv[1]
    lib = sys.argv[2]
    registry = introspect_binary_tool(binary, lib)
    print(json.dumps(registry, indent=2))
