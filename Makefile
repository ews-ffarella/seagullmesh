OS := $(shell uname -s)
# Ensure `make` with no arguments runs the `all` target by default
.DEFAULT_GOAL := wheel
# Default to all available processors; user can override by setting NPROCS in the
# environment or on the make command line: `make NPROCS=4 test-cpp`.
NPROCS ?= $(shell nproc)
CGAL_INC = /home/ews-ffarella/extern/CGAL-6.1.1/include

.PHONY: wheel
wheel:
	CGAL_INC=$(CGAL_INC) ../../.venv/bin/python setup.py build_ext --inplace --parallel=$(NPROCS) 
	CGAL_INC=$(CGAL_INC) ../../.venv/bin/python setup.py bdist_wheel	
	@ls -lah ./dist/
