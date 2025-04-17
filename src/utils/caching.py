"""
Caching utilities for the DomainFusion framework.

This module provides functions and classes for caching results to
improve performance and reduce API calls.
"""

import os
import json
import pickle
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from functools import wraps
import threading
import sqlite3
from datetime import datetime, timedelta


class MemoryCache:
    """
    Simple in-memory cache with optional time-based expiration.
    """
    
    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        """
        Initialize an in-memory cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
            ttl: Time-to-live in seconds (None for no expiration)
        """
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.RLock()
    
    def _compute_key(self, obj: Any) -> str:
        """
        Compute a cache key for an object.
        
        Args:
            obj: Object to compute key for
            
        Returns:
            String key
        """
        try:
            # Try to use json for serialization
            key = json.dumps(obj, sort_keys=True)
        except (TypeError, ValueError, OverflowError):
            # Fall back to pickle and hash
            key = hashlib.md5(pickle.dumps(obj)).hexdigest()
        
        return key
    
    def get(self, key: Any, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value to return if key not found
            
        Returns:
            Cached value or default
        """
        with self.lock:
            key_str = self._compute_key(key)
            
            if key_str in self.cache:
                value, timestamp = self.cache[key_str]
                
                # Check if expired
                if self.ttl is not None and time.time() - timestamp > self.ttl:
                    del self.cache[key_str]
                    return default
                
                return value
            
            return default
    
    def set(self, key: Any, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            key_str = self._compute_key(key)
            
            # Evict oldest item if at capacity
            if len(self.cache) >= self.max_size and key_str not in self.cache:
                # Find oldest item
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            # Store value with timestamp
            self.cache[key_str] = (value, time.time())
    
    def delete(self, key: Any) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was found and deleted, False otherwise
        """
        with self.lock:
            key_str = self._compute_key(key)
            
            if key_str in self.cache:
                del self.cache[key_str]
                return True
            
            return False
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Number of items in the cache
        """
        with self.lock:
            return len(self.cache)


class DiskCache:
    """
    Persistent disk-based cache using SQLite.
    """
    
    def __init__(self, cache_dir: str = ".cache", db_name: str = "domainfusion_cache.db", 
                ttl: Optional[int] = None):
        """
        Initialize a disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            db_name: Name of the SQLite database file
            ttl: Time-to-live in seconds (None for no expiration)
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, db_name)
        self.ttl = ttl
        self.lock = threading.RLock()
        
        # Initialize database
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL
                )
            ''')
            conn.commit()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection.
        
        Returns:
            SQLite connection
        """
        return sqlite3.connect(self.db_path)
    
    def _compute_key(self, obj: Any) -> str:
        """
        Compute a cache key for an object.
        
        Args:
            obj: Object to compute key for
            
        Returns:
            String key
        """
        try:
            # Try to use json for serialization
            key = json.dumps(obj, sort_keys=True)
        except (TypeError, ValueError, OverflowError):
            # Fall back to pickle and hash
            key = hashlib.md5(pickle.dumps(obj)).hexdigest()
        
        return key
    
    def get(self, key: Any, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value to return if key not found
            
        Returns:
            Cached value or default
        """
        with self.lock:
            key_str = self._compute_key(key)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value, timestamp FROM cache WHERE key = ?", (key_str,))
                result = cursor.fetchone()
                
                if result:
                    value_blob, timestamp = result
                    
                    # Check if expired
                    if self.ttl is not None and time.time() - timestamp > self.ttl:
                        cursor.execute("DELETE FROM cache WHERE key = ?", (key_str,))
                        conn.commit()
                        return default
                    
                    try:
                        return pickle.loads(value_blob)
                    except (pickle.PickleError, EOFError):
                        return default
                
                return default
    
    def set(self, key: Any, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            key_str = self._compute_key(key)
            value_blob = pickle.dumps(value)
            timestamp = time.time()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, ?)",
                    (key_str, value_blob, timestamp)
                )
                conn.commit()
    
    def delete(self, key: Any) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was found and deleted, False otherwise
        """
        with self.lock:
            key_str = self._compute_key(key)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache WHERE key = ?", (key_str,))
                deleted = cursor.rowcount > 0
                conn.commit()
                return deleted
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache")
                conn.commit()
    
    def size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Number of items in the cache
        """
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM cache")
                return cursor.fetchone()[0]
    
    def cleanup_expired(self) -> int:
        """
        Remove expired items from the cache.
        
        Returns:
            Number of items removed
        """
        if self.ttl is None:
            return 0
        
        with self.lock:
            expiration_time = time.time() - self.ttl
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache WHERE timestamp < ?", (expiration_time,))
                deleted = cursor.rowcount
                conn.commit()
                return deleted


def cached(cache_instance: Union[MemoryCache, DiskCache] = None, 
          key_fn: Optional[Callable[..., Any]] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache_instance: Cache instance to use (creates a MemoryCache if None)
        key_fn: Optional function to compute cache key from function arguments
        
    Returns:
        Decorated function
    """
    if cache_instance is None:
        cache_instance = MemoryCache()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Compute cache key
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                key = (func.__name__, args, tuple(sorted(kwargs.items())))
            
            # Check cache
            cached_result = cache_instance.get(key)
            if cached_result is not None:
                return cached_result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache_instance.set(key, result)
            return result
        
        return wrapper
    
    return decorator


def get_memory_cache(max_size: int = 1000, ttl: Optional[int] = None) -> MemoryCache:
    """
    Get a memory cache instance.
    
    Args:
        max_size: Maximum number of items to store in the cache
        ttl: Time-to-live in seconds (None for no expiration)
        
    Returns:
        MemoryCache instance
    """
    return MemoryCache(max_size=max_size, ttl=ttl)


def get_disk_cache(cache_dir: str = ".cache", 
                  db_name: str = "domainfusion_cache.db",
                  ttl: Optional[int] = None) -> DiskCache:
    """
    Get a disk cache instance.
    
    Args:
        cache_dir: Directory to store cache files
        db_name: Name of the SQLite database file
        ttl: Time-to-live in seconds (None for no expiration)
        
    Returns:
        DiskCache instance
    """
    return DiskCache(cache_dir=cache_dir, db_name=db_name, ttl=ttl)
