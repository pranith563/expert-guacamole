import argparse
from dataclasses import dataclass, asdict
from typing import Any, List, Optional


@dataclass
class ArgumentSpec:
    flags: List[str]
    dest: str
    required: bool
    default: Any
    action: str
    nargs: Any
    choices: Optional[List[Any]]
    metavar: Optional[str]
    help: Optional[str]
    parser: str
    group: Optional[str] = None


class ArgparseRegistry:
    def __init__(self):
        self.arguments: List[ArgumentSpec] = []

    def capture(self, parser: argparse.ArgumentParser):
        self.arguments.clear()
        
        # Track which action belongs to which group
        action_to_group = {}
        for group in parser._action_groups:
            group_name = group.title
            for action in group._group_actions:
                action_to_group[id(action)] = group_name

        # Track mutually exclusive groups
        mutex_groups = {}
        for i, group in enumerate(parser._mutually_exclusive_groups):
            group_id = f"mutex_{i}"
            for action in group._group_actions:
                mutex_groups[id(action)] = group_id

        for action in parser._actions:
            # Skip suppressed arguments to match --help
            if action.help == '==SUPPRESS==':
                continue
            
            # Skip help action as it's standard
            if isinstance(action, argparse._HelpAction) and not action.option_strings: # positional help? unlikely but safe
                continue
            if "-h" in action.option_strings or "--help" in action.option_strings:
                continue

            group = action_to_group.get(id(action))
            mutex = mutex_groups.get(id(action))
            
            self.arguments.append(ArgumentSpec(
                flags=list(action.option_strings),
                dest=action.dest,
                required=getattr(action, "required", False),
                default=action.default,
                action=action.__class__.__name__,
                nargs=action.nargs,
                choices=list(action.choices) if action.choices else None,
                metavar=action.metavar,
                help=action.help,
                parser=parser.prog,
                group=mutex if mutex else group,
            ))

    def clear(self):
        self.arguments.clear()

    def dump(self):
        def serialize(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            if isinstance(obj, dict):
                return {str(k): serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [serialize(x) for x in obj]
            # Handle classes, enums, etc by converting to string
            return str(obj)

        return [serialize(asdict(a)) for a in self.arguments]


REGISTRY = ArgparseRegistry()


def install():
    pass


def uninstall():
    pass
