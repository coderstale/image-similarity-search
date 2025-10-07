CC = gcc

CFLAGS = -g -Wall -Iinclude

SRCDIR = src

OBJDIR = obj

TARGET = bin/search_app

SOURCES = $(wildcard $(SRCDIR)/*.c)

OBJECTS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SOURCES))

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) bin

.PHONY: all clean