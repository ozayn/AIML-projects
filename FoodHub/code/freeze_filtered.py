
import pkg_resources

# Load the original package list (no versions)
with open("requirements0.txt") as f:
    wanted = {line.strip().lower() for line in f if line.strip() and not line.startswith("#")}

# Get installed distributions
installed = {dist.key: f"{dist.project_name}=={dist.version}" for dist in pkg_resources.working_set}

# Match and save only the ones in your original list
with open("requirements.txt", "w") as out:
    for name in sorted(wanted):
        match = installed.get(name.lower())
        if match:
            out.write(match + "\n")
