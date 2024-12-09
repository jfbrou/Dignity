# Import libraries
import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_script(script_path, cwd, project_root):
    """Run a Python script as a subprocess with a specified working directory and PYTHONPATH."""
    try:
        # Modify the environment to include project_root in PYTHONPATH
        env = os.environ.copy()
        existing_pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = os.pathsep.join([project_root, existing_pythonpath])
        
        logging.info(f"Running {script_path} in {cwd} with PYTHONPATH={env['PYTHONPATH']}...")
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=cwd,  # Set the working directory
            env=env,  # Modified environment
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"Output of {script_path}:\n{result.stdout}")
        logging.info(f"{script_path} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_path}:")
        logging.error(e.stderr)
        sys.exit(1)

def main():
    # Define the root directory of your project
    project_root = os.getcwd()  # Use current working directory
    
    # Define the sequence of scripts to run with their paths relative to project_root
    scripts = [
        os.path.join('Analysis', 'figures.py'),
        os.path.join('Analysis', 'tables.py')
    ]

    # Run each script with the project_root as the working directory
    for script in scripts:
        script_full_path = os.path.join(project_root, script)
        if not os.path.isfile(script_full_path):
            logging.error(f"Script not found: {script_full_path}")
            sys.exit(1)
        run_script(script_full_path, project_root, project_root)

    logging.info("All analysis scripts executed successfully.")

if __name__ == "__main__":
    main()