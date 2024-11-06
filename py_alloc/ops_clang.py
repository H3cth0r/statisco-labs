from typing import Optional, Union, Dict, Any, List, Callable
import os, sqlite3, contextlib, pickle, tempfile, ctypes, subprocess, pathlib, platform, time
from source_code import BufferOptions, ImageDType, PtrDType

def getenv(key: str, default=0): 
    """
    Retrieves an environment variables value, casting it to the same type as
    default.

    Parameters:
    - Key (str): The name of the environment variable.
    - default (Any): default value to return if the variable is not set.

    Returns:
    - Any: Value of the environment variable of the default.
    """
    return type(default)(os.getenv(key, default))

# Check if the platform is macOs
OSX = platform.system() == "Darwin"

# Determine cache directory and database settings
_cache_dir: str = getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache"))
CACHEDB: str = getenv("CACHEDB", os.path.abspath(os.path.join(_cache_dir, "tinygrad", "cache.db")))
CACHELEVEL = getenv("CACHELEVEL", 2)

VERSION = 16
def db_connection():
    """
    Creates and returns a connection to the SQlite cache database.

    If the database connection does not exist, it initializes it with WAL journal mode.

    Returns:
    - sqlite3.connection: the database connection
    """
    global _db_connection
    if _db_connection is None:
        os.makedirs(CACHEDB.rsplit(os.sep, 1)[0], exist_ok=True)
        _db_connection = sqlite3.connect(CACHEDB, timeout=60, isolation_level="IMMEDIATE")
        with contextlib.suppress(sqlite3.OperationalError): _db_connection.execute("PRAGMA journal_mode=WAL").fetchone()
        # if DEBUG >= 7: _db_connection.set_trace_callback(print)
    return _db_connection

def diskcache_clear():
    """ Clears all tables from the disk cache by dropping them. """
    cur = db_connection().cursor()
    drop_tables = cur.execute("SELECT 'DROP TABLE IF EXISTS ' || quote(name) || ';' FROM sqlite_master WHERE type = 'table';").fetchall()
    cur.executescript("\n".join([s[0] for s in drop_tables]))

def diskcache_get(table:str, key:Union[Dict, str, int]) -> Any:
    """ 
    Retrieves a cached item from the specified table using the given key.

    Parameters:
    - table (str): The anem of the table in the database.
    - key (Union[Dict, str, int]): The key to identify the cached item.
    
    Returns:
    - Any: The cached item or None if not found or if caching is disabled.
    """
    if CACHELEVEL == 0: return None
    if isinstance(key, (str, int)): key = {"key" : key}
    conn = db_connection()
    cur = conn.cursor()
    try: 
        res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))
    except sqlite3.OperationalError: return None
    if (val := res.fetchone()) is not None: return pickle.loads(val[0])
    return None

_db_tables = set()
def diskcache_put(table: str, key: Union[Dict, str, int], val: Any):
    """
    Inserts or updates a cached item in the specified table.

    Parameters:
        table (str): The name of the table in the database.
        key (Union[Dict, str, int]): The key to identify the item.
        val (Any): The value to store in the cache.

    Returns:
        Any: The value that was cached.
    """
    if CACHELEVEL == 0: return val
    if isinstance(key, (str, int)): key = {"key" : key}
    conn = db_connection()
    cur = conn.cursor()
    if table not in _db_tables:
        TYPES = {str: "text", bool: "integer", int: "integer", float: "numeric", bytes: "blob"}
        ltypes = ', '.join(f"{k} {TYPES[type(key[k])]}" for k in key.keys())
        cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}_{VERSION}' ({ltypes}, val blob, PRIMARY KEY ({', '.join(key.keys())}))")
        _db_tables.add(table)
    cur.execute(f"REPLACE INTO '{table}_{VERSION}' ({', '.join(key.keys())}, val) VALUES ({', '.join(['?']*len(key.keys()))}, ?)", tuple(key.values()) + (pickle.dumps(val), ))  # noqa: E501
    conn.commit()
    cur.close()
    return val


class Compiler:
    """
    Base class for compilers that supports optional caching.
    
    Attributes:
        cachekey (Optional[str]): The key used for caching the compiled result.
    """
    def __init__(self, cachekey: Optional[str]=None): self.cachekey = None if getenv("DISABLE_COMPILER_CACHE") else cachekey
    def compile(self, src: str) -> bytes: 
        """
        Compiles source code into a byte representation.

        Parameters:
            src (str): The source code to compile.

        Returns:
            bytes: The compiled bytecode.
        """
        return src.encode()
    def compile_cached(self, src: str) -> bytes:
        """
        Compiles source code, using cached results if available.

        Parameters:
            src (str): The source code to compile.

        Returns:
            bytes: The compiled bytecode, either retrieved from cache or newly compiled.
        """
        if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
            assert not getenv("ASSERT_COMPILE"), f"tried to compile with ASSERT_COMPILE set\n{src}"
            lib = self.compile(src)
            if self.cachekey is not None: diskcache_put(self.cachekey, src, lib)
        return lib

class ClangCompiler(Compiler):
    """
    A compiler class that compiles C source code using Clang.
    
    Attributes:
        args (List[str]): List of arguments for Clang.
    """
    def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None): 
        self.args = ["-march=native"] if args is None else args
        super().__init__(cachekey)
    def compile(self, src:str) -> bytes:
        """
        Compiles C source code to a shared library using Clang.

        Parameters:
            src (str): The C source code.

        Returns:
            bytes: The compiled shared library as bytes.
        """
        with tempfile.NamedTemporaryFile(delete=True) as output_file:
            subprocess.check_output(['gcc', '-shared', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib', 
                                     '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
            return pathlib.Path(output_file.name).read_bytes()

def cpu_time_execution(cb, enable):
    """
    Measures the time taken to execute a callback function if enabled.

    Parameters:
        cb (Callable): The callback function to execute.
        enable (bool): Whether to measure time.

    Returns:
        Optional[float]: The execution time in seconds if `enable` is `True`, otherwise `None`.
    """
    if enable: st = time.perf_counter()
    cb()
    if enable: return time.perf_counter()-st

class ClangProgram:
    """
    Represents a compiled Clang program, loaded as a shared library.

    Attributes:
        name (str): The name of the function to execute.
        lib (bytes): The compiled library as bytes.
    """
    def __init__(self, name:str, lib:bytes):
        # if DEBUG >= 6: cpu_objdump(lib)
        self.name, self.lib = name, lib
        with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
            pathlib.Path(cached_file_path.name).write_bytes(lib)
            self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]
    def __call__(self, *bufs, vals=(), wait=False): 
        """
        Executes the loaded function with provided arguments.

        Parameters:
            bufs (tuple): Buffers to pass to the function.
            vals (tuple): Additional values to pass to the function.
            wait (bool): Whether to measure execution time.

        Returns:
            Optional[float]: Execution time if `wait` is `True`, otherwise `None`.
        """
        return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class Buffer:
    def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options=Optional[BufferOptions]=None,
                 initial_value=Optional[bytes]=None, lb_refcount=0, base:Optional[Buffer]=None, offset:int=0, preallocate=False):
        if isinstance(dtype, ImageDType): options = BufferOptions(image=dtype)
        else: assert isinstance(dtype, DType) and not isinstance(dtype, PtrDType)
        self.device, self.size, self.dtype, self.options, self.offset = device, size, dtype, options, offset
        if base is None:
            assert offset == 0, "base buffers can't have offset"
            self._base = None
            self._lb_refcount = lb_refcount
            if opaque is not None: self.allocate(opaque)
            if initial_value is not None:
                self.allocate()
                self.copyin(memoryview(initial_value))
        else:
            assert base._base is None, "base can't have a base"
            assert device == base.device, "base must have the same device"
            self._base = base
        if preallocate: self. allocate()

# class TensorCore:
#     dims: Tuple[int, int, int]
#     dtype_in: DType
#     dtype_out: DType
#     threads: List[Tuple[int, int]]
#     reduce_axes: List[Tuple[int, int]]
#     @property
#     def early_upcast_axes(self) -> List[Tuple[int, int]]:
#         return [(d,self.dims[d]//sz) for d,sz in [(dim,prod(sz for d,sz in self.threads if d==dim)) for dim in range(2)] if self.dims[d]>sz]
#     upcast_axes: Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]]] # list of (TC dim,amt) that upcast A, B and C
#     st1_pattern: Optional[Tuple[Tuple[Tuple[int,int], ...], Tuple[Tuple[int,int], ...]]] = None # pattern to fix shapetracker for A
#     st2_pattern: Optional[Tuple[Tuple[Tuple[int,int], ...], Tuple[Tuple[int,int], ...]]] = None # pattern to fix shapetracker for B
#     expanded_shape: Optional[Tuple[int, ...]] = None
#     opts_seq: Tuple[str,str] = ("UP","LC") # upcast input, local the thread pattern
#     def __str__(self): return "_".join(["WMMA"] + list(map(str, self.dims)) + [self.dtype_in.name, self.dtype_out.name])
# class Renderer:
#     device: str = ""
#     suffix: str = ""
#     supports_float4: bool = True
#     has_local: bool = True
#     has_shared: bool = True
#     global_max: Optional[Tuple[int, ...]] = (0x8FFFFFFF,) * (3)
#     local_max: Optional[Tuple[int, ...]] = (0x8FFFFFFF,) * (3)
#     shared_max: int = 32768
#     tensor_cores: List[TensorCore] = []
#     extra_matcher: Any = None
#     code_for_op: Dict[op, Callable] = {}
# class Compiled:
#     def __init__(self, device:str, allocator:Allocator, renderer:Optional[Renderer], compiler:Optional[Compiler], runtime, graph=None):
#         self.dname, self.allocator, self.compiler, self.runtime, self.graph = device, allocator, compile or Compiler(), runtime, graph
#         self.renderer = renderer or Renderer
#     def synchronize(self): pass
# class ClangDevice(Compiled):
#     def __init__(self, device:str):
#         # tinygrad.runtime.graph.clang import ClangGraph
#         super().__init__(device, MallocAllocator, Cla)
