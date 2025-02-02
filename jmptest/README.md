

I am trying to modify a python program to be able to communicate with a C++ program using shared memory. The main responsibility of the python program is to read some video frames from an input queue located in shared memory, do something on the video frame and write it back to the output queue in shared memory.

I believe there are few things I need to achieve and it would be great if someone can shed some light on it:

    Shared memory: In C/C++, you can use functions like shmget and shmat to get the pointer to the shared memory. What is the equivalent way to handle this in python so both python and C++ program can use the same piece of shared memory?

    Synchronization: Because this involves multi-processing, we need some sort of locking mechanism for the shared memory in both C++ and python programs. How can I do this in python?

Many thanks!


Perhaps shmget and shmat are not necessarily the most appropriate interfaces for you to be using. In a project I work on, we provide access to a daemon via a C and Python API using memory mapped files, which gives us a very fast way of accessing data

The order of operations goes somewhat like this:

    the client makes a door_call() to tell the daemon to create a shared memory region
    the daemon securely creates a temporary file
    the daemon open()s and then mmap()s that file
    the daemon passes the file descriptor back to the client via door_return()
    the client mmap()s the file descriptor and associates consecutively-placed variables in a structure with that fd
    the client does whatever operations it needs on those variables - when it needs to do so.
    the daemon reads from the shared region and does its own updates (in our case, writes values from that shared region to a log file).

Our clients make use of a library to handle the first 5 steps above; the library comes with Python wrappers using ctypes to expose exactly which functions and data types are needed.

For your problem space, if it's just the python app which writes to your output queue then you can track which frames have been processed just in the python app. If both your python and c++ apps are writing to the output queue then that increases your level of difficulty and perhaps refactoring the overall application architecture would be a good investment.

Thanks for your suggestions. Please can you briefly explain how can I implement the door_call() and door_return() myself as apparently they are Oracle's function? Is it a kind of signal, unix socket or something else? – 
HeiHei
Commented Mar 6, 2018 at 1:55
I spent some time to have a look on CPython and found that we can directly call python functions directly using Py_xxx. Would it be more efficient to pass a video frame directly to Py_xxx(frame) as an argument rather than using shared memory, and obtained the processed frame from the return value of this function? Or any drawback on this? – 
HeiHei
Commented Mar 6, 2018 at 5:00
Re your question about door_call() and door_return(): I'm uncertain whether those functions have been fully implemented for linux; rampant.org/doors/index.html is a bit out of date. You might be better off using unix domain sockets instead. – 
James McPherson
Commented Mar 6, 2018 at 6:12

    CPython might well provide you with the functionality you need. You would need to assess just how efficient (and elegant, frankly) it is given your use-case and functionality requirements. – 
    James McPherson
    Commented Mar 6, 2018 at 6:13




Sorta, kinda shared memory. So not exactly what the OP wanted.
This works using memory mapped files. I do not claim high speed or efficiency in any way. These are just to show an example of it working.

```
 $ python --version
 Python 3.7.9

 $ g++ --version
 g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
```

The C++ side only monitors the values it needs. The Python side only provides the values.
Note: the file name "pods.txt" must be the same in the C++ and python code.
```
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
 
int main(void)
  {
  // assume file exists
  int fd = -1;
  if ((fd = open("pods.txt", O_RDWR, 0)) == -1)
     {
     printf("unable to open pods.txt\n");
     return 0;
     }
  // open the file in shared memory
  char* shared = (char*) mmap(NULL, 8, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  // periodically read the file contents
  while (true)
      {
      printf("0x%02X 0x%02X 0x%02X 0x%02X 0x%02X 0x%02X 0x%02X 0x%02X\n", shared[0], shared[1], shared[2], shared[3], shared[4], shared[5],           shared[6], shared[7]);
      sleep(1);
      }

   return 0;
   }
```

The python side:
```
import os
import time
 
fname = './pods.txt'
if not os.path.isfile(fname):
    # create initial file
    with open(fname, "w+b") as fd:
         fd.write(b'\x01\x00\x00\x00\x00\x00\x00\x00')

# at this point, file exists, so memory map it
with open(fname, "r+b") as fd:
    mm = mmap.mmap(fd.fileno(), 8, access=mmap.ACCESS_WRITE, offset=0)

    # set one of the pods to true (== 0x01) all the rest to false
    posn = 0
    while True:
         print(f'writing posn:{posn}')

         # reset to the start of the file
         mm.seek(0)
 
         # write the true/false values, only one is true
         for count in range(8):
             curr = b'\x01' if count == posn else b'\x00'
             mm.write(curr)

         # admire the view
         time.sleep(2)

         # set up for the next position in the next loop
        posn = (posn + 1) % 8

    mm.close()
    fd.close()
```

To run it, in terminal #1:
``` 
a.out  # or whatever you called the C++ executable
 0x00 0x00 0x00 0x00 0x01 0x00 0x00 0x00
 0x00 0x00 0x00 0x00 0x01 0x00 0x00 0x00
 0x01 0x00 0x00 0x00 0x00 0x00 0x00 0x00
 0x01 0x00 0x00 0x00 0x00 0x00 0x00 0x00
 0x00 0x01 0x00 0x00 0x00 0x00 0x00 0x00
 0x00 0x01 0x00 0x00 0x00 0x00 0x00 0x00
 0x00 0x00 0x01 0x00 0x00 0x00 0x00 0x00
 0x00 0x00 0x01 0x00 0x00 0x00 0x00 0x00
 0x00 0x00 0x00 0x01 0x00 0x00 0x00 0x00
``` 

i.e. you should see the 0x01 move one step every couple of seconds because of the sleep(2) in the C++ code.
in terminal #2:
```
python my.py  # or whatever you called the python file
writing posn:0
writing posn:1
writing posn:2
```
i.e. you should see the position change from 0 through 7 back to 0 again.

https://docs.python.org/3.10/library/multiprocessing.shared_memory.html
