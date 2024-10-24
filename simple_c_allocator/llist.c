#include "llist.h";

void add_node(bin_t *bin, node_t* node) {
    node->next = NULL;
    node->prev = NULL;

    node_t *temp = bin->head;

    if (bin->head == NULL) {
        bin->head = node;
        return;
    }

    node_t *current   = bin->head;
    node_t *previous  = NULL;

    while (current  != NULL && current->size <= node->size) {
        previous  = current;
        current   = current->next;
    }

    if (current == NULL) {
        previous->next  = node;
        node->prev      = previous;
    } else {
        if (previous != NULL) {
            node->next      = current;
            previous->next  = node;

            node->prev    = previous;
            current->prev = node;
        } else {
            node->next      = bin->head;
            bin->head->prev = node;
            bin->head       = node;
        }
    }
}

void remove_node(bin_t *bin, node_t *node) {
    if (bin->head == NULL) return;
    if (bin->head == node) {
        bin->head = bin->head->next;
        return;
    }

    node_t *temp = bin->head->next;
    while (temp != NULL) {
        if (temp == node) {
            if (temp->next == NULL) {
                temp->prev->next = NULL;
            } else {
                temp->prev->next = temp->next;
                temp->next->prev = temp->prev;
            }
            return;
        }
        temp = temp->next;
    }
}

node_t *get_best_fit(bin_t *bin, size_t size) {
    if (bin->head == NULL) return NULL; // list is empty

    node_t *temp = bin->head;

    while (temp != NULL) {
        if (temp->size >= size) return temp;
        temp = temp->next;
    }
    return NULL; // no fit
}
node_t *get_last_node(bin_t *bin) {
    node_t *temp = bin->head;
    while (temp->next != NULL) { temp = temp->next; }
    return temp;
}
