from typing import Optional, Union, Dict, Any, List
import os, sqlite3, contextlib, pickle, tempfile, ctypes, subprocess, pathlib, platform

def getenv(key: str, default=0): return type(default)(os.getenv(key, default))

OSX = platform.system() == "Darwin"

_cache_dir: str = getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache"))
CACHEDB: str = getenv("CACHEDB", os.path.abspath(os.path.join(_cache_dir, "tinygrad", "cache.db")))
CACHELEVEL = getenv("CACHELEVEL", 2)

VERSION = 16
def db_connection():
    global _db_connection
    if _db_connection is None:
        os.makedirs(CACHEDB.rsplit(os.sep, 1)[0], exist_ok=True)
        _db_connection = sqlite3.connect(CACHEDB, timeout=60, isolation_level="IMMEDIATE")
        with contextlib.suppress(sqlite3.OperationalError): _db_connection.execute("PRAGMA journal_mode=WAL").fetchone()
        # if DEBUG >= 7: _db_connection.set_trace_callback(print)
    return _db_connection

def diskcache_clear():
    cur = db_connection().cursor()
    drop_tables = cur.execute("SELECT 'DROP TABLE IF EXISTS ' || quote(name) || ';' FROM sqlite_master WHERE type = 'table';").fetchall()
    cur.executescript("\n".join([s[0] for s in drop_tables]))

def diskcache_get(table:str, key:Union[Dict, str, int]) -> Any:
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
    def __init__(self, cachekey: Optional[str]=None): self.cachekey = None if getenv("DISABLE_COMPILER_CACHE") else cachekey
    def compile(self, src: str) -> bytes: return src.encode()
    def compile_cached(self, src: str) -> bytes:
        if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
            assert not getenv("ASSERT_COMPILE"), f"tried to compile with ASSERT_COMPILE set\n{src}"
            lib = self.compile(src)
            if self.cachekey is not None: diskcache_put(self.cachekey, src, lib)
        return lib

class ClangCompiler(Compiler):
    def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None): 
        self.args = ["-march=native"] if args is None else args
        super().__init__(cachekey)
    def compile(self, src:str) -> bytes:
        with tempfile.NamedTemporaryFile(delete=True) as output_file:
            subprocess.check_output(['gcc', '-shared', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib', 
                                     '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
            return pathlib.Path(output_file.name).read_bytes()
