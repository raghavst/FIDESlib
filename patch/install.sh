#!/usr/bin/env bash

# This script is used to install the OpenFHE library.

# Check arguments.
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
	echo "Usage: $0 <path-to-OpenFHE-source> [<absolute-path-to-installation-location>]"
	exit 1
fi

# Extract the source and installation paths.
SOURCE_PATH=$1
if [ $# -eq 2 ]; then
	INSTALL_PATH=$2
fi

# Check if the given path is a directory.
if [ ! -d "$SOURCE_PATH" ]; then
	echo "Error: $SOURCE_PATH is not a directory"
	exit 2
fi

# Check if the given installation path is a directory.
if [ $# -eq 2 ] && [ ! -d "$INSTALL_PATH" ]; then
	echo "Error: $INSTALL_PATH is not a directory"
	exit 3
fi

# Check if the given directory contains the OpenFHE git repository
if [ ! -d "$SOURCE_PATH"/.git ]; then
	echo "Error: $SOURCE_PATH is not a git repository"
	exit 4
fi

# Build the OpenFHE library and install it
cd "$SOURCE_PATH" || exit 5
if [ ! -d build ]; then
	mkdir build
fi
cd build || exit 6
if [ $# -eq 1 ]; then
	cmake ..
else
	cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH"
fi
make install -j
