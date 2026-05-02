/*
 * lib9.h - Plan 9/Inferno C library interface
 * This header provides the basic types and interfaces for the Inferno system.
 */

#ifndef LIB9_H
#define LIB9_H

/* Need va_list for kern.h function declarations */
#include <stdarg.h>
/* Need jmp_buf and setjmp/longjmp */
#include <setjmp.h>

/* Basic Plan 9 types - must be defined before including kern.h */
#define nil ((void*)0)
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned int u32int;
typedef unsigned char u8int;
typedef unsigned short u16int;
typedef unsigned long long u64int;
typedef unsigned int mpdigit;	/* for libmp/mp.h */
typedef signed char schar;
typedef long long vlong;
typedef unsigned long long uvlong;
typedef unsigned long uintptr;
typedef uint Rune;

/* Include the full kernel/system interface which provides all constants and functions */
#include <kern.h>

#endif /* LIB9_H */