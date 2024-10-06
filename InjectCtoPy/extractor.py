from subprocess import call

path = "./build/temp.linux-x86_64-cpython-310/text.o"
with open(path, "rb") as fil:
    with open("text.py", "w") as ut:
        ut.write(str(fil.read()))
