from subprocess import call

path_o = "./build/temp.linux-x86_64-cpython-310/simpleops.o"

call(["python", "setup.py", "build_ext", "--inplace"])
call(["cp", path_o, "./"])
call(["objcopy", "--only-section=.text", "-O", "binary", "simpleops.o", "text.o"])

with open("text.o", "rb") as fil:
    with open("text.py", "w") as ut:
        ut.write("code=")
        ut.write(str(fil.read()))
print("Done setup")
