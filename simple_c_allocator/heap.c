#include "heap.h"

uint offset = 9;

// ========================================================
// This function initializes a new heap structure, provides
// an empty heap struct, and a place to start the heap
//
// This function uses HEAP_INIT_SIZE to determine how large
// the heap is to make sure the same constant is used when 
// allocating memory for your heap
// ========================================================
void init_heap(heap* heap, long start) {
    // Creating a initial region, this is the "wilderness" chunk
    // The heap starts as just one big chunk of allocatable memory
    node_t* init_region   = (node_t *) start;
    init_region->hole     = 1;
    init_region->size     = (HEAP_INIT_SIZE) - sizeof(node_t) - sizeof(footer_t);

    // Create a foot (size must be defined)
    create_foot(init_region);

    // add the region to the correct bin and setup the heap struct
    add_node(heap->bins[get_bin_index(init_region->size)], init_region);

    heap->start           = (void *) start;
    heap->end             = (void *) (start + HEAP_INIT_SIZE);
}

// ========================================================
// Allocation function of the heap, takes the heap struct
// pointer and the size of the chunk we want. This function
// will search through the bins until it finds a suitable 
// chunk. It will then split the chunk if necessary and
// return the start of the chunk.
// ========================================================
void *heap_alloc(heap_t *heap, size_t size) {
    // First get the bin index where this chunk size should be in
    uint index     = get_bin_index(size);

    // Use this bin to try and find a good fitting chunk
    bin_t *temp   = (bin_t *) heap->bins[index];
    node_t *found = get_best_fit(temp, size);

    // While no chunk is found advance through the bins until
    // find a chunk or get to the wilderness
    while (found == NULL) {
        if (index + 1 >= BIN_COUNT) return NULL;
        temp      = heap->bins[++index];
        found     = get_best_fit(temp, size);
    }

    // If the difference between the found chunk and the requested
    // chunk is bigger than the overhead (metadata size) + the min
    // alloc size then we should split this chunk, otherwise just 
    // return the chunk
    if ((found->size - size) > (overhead + MIN_ALLOC_SZ)) {
        // Do the math where to split at, then set its metadata
        node_t *split   = (node_t *) (((char *) found + sizeof(node_t) + sizeof(footer_t)) + size);
        split->size     = found->size - size - sizeof(node_t) - sizeof(footer_t);
        split->hole     = 1;

        // Create a footer for the split
        create_foot(split);

        // Now we need to get the new index for this split chunk
        // place it in the correct bin
        uint new_idx     = get_bin_index(split->size);
        add_node(heap->bind[new_idx], split);

        found->size     = size; // set the found chunk size
        create_foot(found);     // sinze size changed, remake foot
    }

    found->hole   = 0;                      // not a hole anymore
    remove_node(heap->bins[index], found);  // remove it from its bin

    // Then it checks to determine if the heap should be expanded
    // or contracted
    node_t *wild  = get_wilderness(heap);
    if (wild->size < MIN_WILDERNESS) {
        uint success = expand(heap, 0x1000);
        if (success == 0) return NULL;
    } else if (wild->size > MAX_WILDERNESS) {
        contract(heap, 0x1000);
    }

    // sinze prev and next are not needed fields when the
    // chunk is in use by the user, it can be cleared and 
    // return the address of the next field
    found->prev   = NULL;
    found->next   = NULL;
    return &found->next;
}

// ========================================================
// Takes the heap struct pointer and the pointer provided by
// the heap_alloc function. The given chunk will be possibly
// coalesced and then placed in the correct bin.
// ========================================================
void heap_free(heap_t *heap, void *p) {
    bin_t *list;
    footer_t *new_foot, *old_foot;

    // The actual head of the node is not p, it is p minus the
    // size of the fields that precede "next" in the node 
    // structure if the node being free is the start of the heap
    // then there is no need to coalesce so just it in the right 
    // list
    node_t *head = (node_t *) ((char *) p - offset);
    if (head == (node_t *) (uintptr_t) heap->start) {
        head->hole = 1;
        add_node(heap->bins[get_bin_index(head->size)], head);
        return;
    }

    // These are the next and previous nodes in the heap, not the prev
    // and next in a bin. To find prev we jost get substract from the
    // start of the head node to get the footer of the previous node (which gives us
    // the header pointer). To get the next node we simply get the footer
    // and add the sizeof(footer_t)
    node_t * next   = (node_t *) ((char *) get_foot(head) + sizeof(footer_t));
    footer_t *f     = (footer_t *) ((char *) head - sizeof(footer_t));
    node_t *prev    = f->header;

    // If the previous node is a hole we can coalese
    if (prev->hole) {
        // remove the previous node from its bin
        list = heap->bins[get_bin_index(prev->size)];
        remove_node(list, prev);

        // re-calculate the size of the node and recreate a footer
        prev->size += overhead + head->size;
        new_foot = get_foot(head);
        new_foot->header = prev;

        // previous is now the node we are working with, head to 
        // prev because the next if statement will coalesce with the
        // next node and we want that statement to work even when we
        // coalesce with prev
        heap = prev;
    }

    // if the next node is free coalesce
    if (next->hole) {
        // remove it from its bin
        list = heap->bins[get_bin_index(next->size)];
        remove_node(list, next);

        // re-calculate the new size of head
        head->size += overhead + next->size;

        // clear out the old metadata from next
        old_foot = get_foot(next);
        old_foot->header = 0;
        next->size = 0;
        next->hole = 0;

        new_foot = get_foot(head);
        new_foot->header = head;
    }

    // This chunk is now a hole, then put it in the right bin
    head->hole = 1;
    add_node(heap->bins[get_bin_index(head->size)], head);
}

int expand(heap_t *heap, size_t sz) {
    return 0;
}

void contract(heap_t *heap, size_t sz) {
    return;
}

// ========================================================
// Hashing function that converts size => bin index. Changing
// this function will change the binning policy of the heap.
// Right now it just places any allocation < 8 in bin 0 and 
// then for anything above 8 it bins using the log base 2 of
// the size.
// ========================================================
uint get_bin_index(size_t sz) {
    uint index = 0;
    sz = sz < 4 ? 4 : sz;
    while (sz >>= 1) index ++;
    index -= 2;
    if (index > BIN_MAX_IDX) index = BIN_MAX_IDX;
    return index;
}

// ========================================================
// This function will create a footer given a node the node's
// size must be set to the correct value
// ========================================================
void create_foot(node_t *head) {
    footer_t *foot  = get_foot(head);
    foot->header    = head;
}

// ========================================================
// This function will get the footer pointer given a node
// ========================================================
footer_t *get_foot(node_t *node) {
    return (footer_t *) ((char *) node + sizeof(node_t) + node->size);
}

// ========================================================
// This function will  get the wilderness node given a heap
// struct pointer. THis function banks on the heap's end
// field being correct, it simply uses the footer at the end 
// of the heap because that is always the wilderness
// ========================================================
node_t *get_wilderness(heap_t *heap) {
    footer_t *wild_foot = (footer_t *) ((char *) heap->end - sizeof(footer_t));
    return wild_foot->header;
}
