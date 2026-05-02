#include "lib9.h"
#include <pwd.h>

/* forward declaration to avoid including <unistd.h> which conflicts with kern.h */
extern unsigned int getuid(void);

char*
getuser(void)
{
	struct passwd *p;

	static char *user = 0;

	if (!user) {
		p = getpwuid(getuid());
		if (p && p->pw_name) {
			user = malloc(strlen(p->pw_name)+1);
			if (user)
				strcpy(user, p->pw_name);
		}
	}
	if(!user)
		user = "unknown";
	return user;
}
