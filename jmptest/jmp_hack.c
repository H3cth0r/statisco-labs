#include <stdio.h>
#include <setjmp.h>

jmp_buf env;
int result = 0;

int hack_add(int a, int b) {
    if (setjmp(env) == 0) {
        // First call, save state and return
        printf("Saving CPU state...\n");
        return 0;  // This is ignored when longjmp is used
    } else {
        // Jumped back! Return the modified result
        printf("Jumped back! Returning %d\n", result);
        return result;
    }
}

// Expose longjmp so Python can call it
void modify_and_jump(int new_value) {
    result = new_value;
    longjmp(env, 1);
}
