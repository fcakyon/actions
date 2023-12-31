#!/bin/sh -l

# Format Python code
echo "Running Ruff for Python code formatting..."
ruff . --line-length 120

# Format markdown files
echo "Running mdformat for Markdown formatting..."
mdformat .

# Format Python docstrings
echo "Running docformatter..."
docformatter -i -r .

# Run spell check
echo "Running codespell for spell checking..."
codespell -w

# Commit and push changes
echo "Committing and pushing changes..."
git config --global user.name "glenn-jocher"
git config --global user.email "glenn.jocher@ultralytics.com"
git add -A
git commit -m "Ultralytics format updates" || echo "No changes to commit"
git push
