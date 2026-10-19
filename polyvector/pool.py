import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager

class ConnectionPool:
    """
    Thread-safe persistent connection pool for PolyVector.
    Eliminates TCP connection handshakes on search queries.
    """
    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str]] = None,
        minconn: int = 4,
        maxconn: int = 30
    ):
        self.minconn = minconn
        self.maxconn = maxconn
        self.config = config or {}
        self._pool: Optional[ThreadedConnectionPool] = None
        self._init_pool()

    def _init_pool(self):
        if isinstance(self.config, str):
            # Connection URI string
            self._pool = ThreadedConnectionPool(
                minconn=self.minconn,
                maxconn=self.maxconn,
                dsn=self.config
            )
        else:
            # Config dictionary
            self._pool = ThreadedConnectionPool(
                minconn=self.minconn,
                maxconn=self.maxconn,
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 5432),
                dbname=self.config.get("dbname", "postgres"),
                user=self.config.get("user", "postgres"),
                password=self.config.get("password", "")
            )

    @contextmanager
    def connection(self):
        if self._pool is None or self._pool.closed:
            self._init_pool()
        conn = self._pool.getconn()
        try:
            yield conn
        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            if conn and not self._pool.closed:
                self._pool.putconn(conn)

    def close(self):
        if self._pool and not self._pool.closed:
            self._pool.closeall()
