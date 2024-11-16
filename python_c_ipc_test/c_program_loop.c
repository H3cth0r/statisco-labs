#include <stdio.h>
#include <string.h>

int main() {
    char operation[10];
    int num1, num2;

    while (1) {
        if (scanf("%s %d %d", operation, &num1, &num2) != 3) {
            break; 
        }

        int result = 0;

        if (strcmp(operation, "add") == 0) {
            result = num1 + num2;
        } else if (strcmp(operation, "sub") == 0) {
            result = num1 - num2;
        } else if (strcmp(operation, "mul") == 0) {
            result = num1 * num2;
        } else {
            printf("Error: Unknown operation\n");
            fflush(stdout);
            continue;
        }

        printf("%d\n", result);
        fflush(stdout);
    }

    return 0;
}
