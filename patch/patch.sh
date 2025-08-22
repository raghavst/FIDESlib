#!/usr/bin/env bash

# This script is used to patch the OpenFHE source code.

# Check if a route is given as argument
if [ $# -gt 1 ]; then
	echo "Usage: $0 [<path-to-OpenFHE-source>]"
	exit 1
fi

# Check if the given route is a directory.
if [ $# -eq 1 ] && [ ! -d "$1" ]; then
	echo "Error: $1 is not a directory"
	exit 2
fi

# Check if the given directory contains the OpenFHE git repository.
if  [ $# -eq 1 ] && [ ! -d "$1"/.git ]; then
	echo "Error: $1 is not a git repository"
	exit 3
fi

# Get the directory where this script is located.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Clone the OpenFHE development repository if needed.
if [ ! -d "$1" ]; then
	git clone https://github.com/openfheorg/openfhe-development.git "$DIR"/openfhe-development
	SOURCE="$DIR"/openfhe-development
else
	SOURCE="$1"
fi

# Apply the patch.
cd "$SOURCE" || exit 4
git am "$DIR"/openfhe-hook.patch
git am "$DIR"/openfhe-base.patch