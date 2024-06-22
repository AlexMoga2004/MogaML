# Compiler and flags
CC = gcc
CFLAGS = -Wall -Werror -Iinclude

# Directories
SRCDIR = src
INCDIR = include
TESTDIR = test
BINDIR = bin

# Source files and object files
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(patsubst $(SRCDIR)/%.c, $(SRCDIR)/%.o, $(SOURCES))
TEST_SOURCES = $(wildcard $(TESTDIR)/*.c)

# Executable names
LIB = libmatrix.a
TEST_EXES = $(patsubst $(TESTDIR)/%.c, $(TESTDIR)/%, $(TEST_SOURCES))

# Targets
.PHONY: all clean test

all: $(BINDIR)/$(LIB) $(TEST_EXES)

# Create the static library
$(BINDIR)/$(LIB): $(OBJECTS)
	@mkdir -p $(BINDIR)
	ar rcs $@ $^

# Compile object files
$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Run tests
test: $(TEST_EXES)
	for test_exe in $(TEST_EXES); do \
		echo "Running $$test_exe"; \
		./$$test_exe; \
	done

# Build test executables
$(TESTDIR)/%: $(TESTDIR)/%.c $(BINDIR)/$(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L$(BINDIR) -lmatrix -lm

clean:
	rm -f $(SRCDIR)/*.o $(BINDIR)/* $(TEST_EXES)
