
import os
import pkg_resources
import inspect
import sys

def run_command(command=""):
    os.system(command)

def create_requirements_with_versions(req_name_orig="requirements0.txt", 
                                      req_name="requirements.txt", quiet=True, 
                                      upgrade=False):
    install_command = f"pip install -r {req_name_orig}"
    if quiet:
        install_command+= " -q"
    if upgrade:
        install_command+= " --upgrade"
    run_command(install_command)
    
    # Load the original package list (no versions)
    with open(req_name_orig) as f:
        wanted = {line.strip().lower() for line in f if line.strip() and not line.startswith("#")}
    
    # Get installed distributions
    installed = {dist.key: f"{dist.project_name}=={dist.version}" for dist in pkg_resources.working_set}
    
    # Match and save only the ones in your original list
    with open(req_name, "w") as out:
        for name in sorted(wanted):
            match = installed.get(name.lower())
            if match:
                out.write(match + "\n")


def is_running_colab():
    return 'google.colab' in sys.modules


def get_function_def(func):
    print(inspect.getsource(func))

def is_same_function(func, ftext):
    out1 = inspect.getsource(func)
    return out1==ftext
