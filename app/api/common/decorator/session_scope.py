from __future__ import annotations

import asyncio
import inspect
from functools import wraps

from app.api.domain.infra.db.session import SessionLocal


def session_scope(func):
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def _awrapper(*args, **kwargs):
            session = SessionLocal()
            try:
                # Prefer 'session' param name
                params = inspect.signature(func).parameters
                if "session" in params and "session" not in kwargs:
                    kwargs["session"] = session
                elif "db" in params and "db" not in kwargs:
                    # Backward compatibility
                    kwargs["db"] = session
                result = await func(*args, **kwargs)
                session.commit()
                return result
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        return _awrapper

    @wraps(func)
    def _swrapper(*args, **kwargs):
        session = SessionLocal()
        try:
            params = inspect.signature(func).parameters
            if "session" in params and "session" not in kwargs:
                kwargs["session"] = session
            elif "db" in params and "db" not in kwargs:
                kwargs["db"] = session
            result = func(*args, **kwargs)
            session.commit()
            return result
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return _swrapper


