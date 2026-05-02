#include "lib9.h"
#include <sys/types.h>
#include <fcntl.h>

/* forward declaration to avoid including <unistd.h> which conflicts with kern.h */
extern long lseek(int, long, int);

vlong
seek(int fd, vlong where, int from)
{
	return lseek(fd, where, from);
}
