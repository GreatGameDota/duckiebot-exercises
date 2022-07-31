
### If running into permission denied errors:

- Open file explorer (Home) and enable viewing of hidden files
- Navigate to `~/.dt-shell/commands-multi/daffy/exercises/test/command.py`
- After line 958 add the line:

```py
agent_container.exec_run(f"chmod +x /code/launchers/{config['agent_run_cmd']}")
```
