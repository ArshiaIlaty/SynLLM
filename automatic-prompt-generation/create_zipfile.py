import shutil
import os

# Make sure you're in the correct directory
os.chdir('/home/jovyan/SynLLM/automatic-prompt-generation')

# Create zip file
shutil.make_archive('prompts', 'zip', '.', 'SynLLM/automatic-prompt-generation/prompts')

# This will create prompts.zip in your home directory
print("Created prompts.zip in your home directory")
